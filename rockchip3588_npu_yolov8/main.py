# (C) Markham Lee 2024
# https://github.com/MarkhamLee/computer-vision-demo-reel
# Post processing loosely based/inspired by:
# https://github.com/MarkhamLee/rknn_model_zoo/blob/main/examples/yolov8/python/yolov8.py
# YOLOv8 object counter running on a Rockchip3588 NPU via the RKNN format.
# For YOLO models the RKNN conversion process pushes
# some of the post processing steps that would normally occur on a GPU
# to the CPU:
# https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.md
# Through the use of asynchronous methods + restricting the number of threads
# used to two, this demo can run at about 30 FPS vs 20 FPS for an older
# approach that didn't limit the threads or use asynchronous methods. Better
# yet, this approach can support two video feeds running at 30+ FPS, provided
# you run each one on a separate NPU  core.
import asyncio
import cv2
import os
import sys
import numpy as np
import rknn_yolov8_config as config
from post_process import PostProcess
from queue import Queue
from rknnlite.api import RKNNLite
from time import time

# this ensures we only use the big cores, to get
# the best performance.
os.environ['OMP_NUM_THREADS'] = '2'

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from common_utils.logging_utils import LoggingUtilities  # noqa: E402

IMG_SIZE = config.IMG_SIZE
CLASSES = config.CLASSES


class RKYoloV8:

    def __init__(self, video_path: str, model_path: str):

        # define queues
        self.pre_processed_images = Queue()
        self.inferencing_data = Queue()
        self.post_process_data = Queue()
        self.prepared_images = Queue()

        self.post_process = PostProcess()
        self.logger = LoggingUtilities.console_out_logger("Main Script")

        # instantiate RKNN API
        self.rknn = RKNNLite()

        self.streaming_object = self.create_streaming_object(video_path)
        self.model = self.load_model(model_path)
        self.logger.info("Model Loaded")

        self.run_time_environment()
        self.logger.info("Run time environment created")
        asyncio.run(self.inferencing())

    def load_model(self, model_path):

        model = self.rknn.load_rknn(model_path)
        if model != 0:
            self.logger.debug("Model load error, exiting....")
            sys.exit()

        return model

    def run_time_environment(self):

        status = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)

        if status != 0:
            self.logger.debug('Run time instantiation failed, exiting...')
            sys.exit()
        else:
            self.logger.info('RKNN run time environment created')

    def create_streaming_object(self, path: str):

        stream_object = cv2.VideoCapture(path)
        assert stream_object.isOpened(), "Error reading video file"

        return stream_object

    async def pre_process(self):

        # read video file or stream
        success, frame = self.streaming_object.read()

        if not success:
            self.logger.info('Data read failure or video over, exiting...')
            self.shutdown()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        shape = frame.shape[:2]

        if shape != IMG_SIZE:
            frame = self.letter_box(frame, shape, (640, 640), (0, 0, 0))

        self.pre_processed_images.put(frame)

        # put the same frame aside to draw the detection boxes on
        self.prepared_images.put(frame)

    def letter_box(self, frame, shape, new_shape, pad_color):

        # Resize and pad image while meeting stride-multiple constraints
        # Scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        # ratio = r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            frame = cv2.resize(frame, new_unpad,
                               interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=pad_color)

        return frame

    async def inferencing(self):

        frame_counter = 0
        total_inferencing_latency = 0
        total_post_process_latency = 0
        total_latency = 0

        while True:

            await self.pre_process()

            # Inference
            start = time()
            await self.inference(self.pre_processed_images.get())
            end = time()

            # calculate inferencing latency
            inferencing_latency = 1000 * round((end - start), 2)
            total_inferencing_latency += inferencing_latency

            start_process_time = time()
            boxes, classes, scores = await self.\
                post(self.inferencing_data.get())
            end_process_time = time()

            post_process_latency = 1000 * round((end_process_time -
                                                 start_process_time), 2)
            total_post_process_latency += post_process_latency
            total_latency = total_post_process_latency +\
                total_inferencing_latency

            frame_counter += 1

            overall_fps = round((1000 / (total_latency / frame_counter)), 2)

            # used for troubleshooting/testing

            if frame_counter % 100 == 0:
                avg_inferencing_latency = round((total_inferencing_latency /
                                                 frame_counter), 2)
                avg_post_latency = round((total_post_process_latency /
                                          frame_counter), 2)
                inference_fps = round((1000 / avg_inferencing_latency), 2)
                self.logger.info(f'Avg inferencing(ms): {avg_inferencing_latency}, inferencing FPS: {inference_fps}, avg post-process(ms): {avg_post_latency}, overall FPS: {overall_fps}')  # noqa: E501

            # draw boxes on image
            await self.draw(self.prepared_images.get(), boxes, classes,
                            scores, overall_fps)

            key = cv2.waitKey(1)
            if key == ord('q'):
                self.shutdown()

    async def inference(self, frame):

        self.inferencing_data.put(self.rknn.
                                  inference
                                  (inputs=[(np.expand_dims(frame, 0))]))

    async def post(self, data):

        return self.post_process.post_process(data)

    async def draw(self, image, boxes, classes, scores, fps):

        message = (f'Overall FPS: {fps}')

        cv2.putText(image, message, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 1)

        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(_b) for _b in box]

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 1)
            cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
            cv2.imshow("YOLOv8 Detections on Orange Pi 5+ w/ Rockchip 3588",
                       image)

    def shutdown(self):

        self.rknn.release()
        cv2.destroyAllWindows()
        self.streaming_object.release()
        sys.exit()


a = RKYoloV8("../videos/cars.mp4",
             "../rknn_models/yolov8n.rknn")
