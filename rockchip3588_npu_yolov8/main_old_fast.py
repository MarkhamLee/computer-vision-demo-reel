# Based on/inspired by:
# https://github.com/MarkhamLee/rknn_model_zoo/blob/main/examples/yolov8/python/yolov8.py
# YOLOv8 object counter running on a Rockchip3588 NPU via the RKNN format.
# For YOLO models the RKNN conversion process pushes
# some of the post processing steps that would normally occur on a GPU
# to the CPU:
# https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.md
# This approach will run about as fast as the newer approach using asynchronous
# methods + restricting the number of threads, however, speed drops
# significantly if you try to run multiple video feeds at once.
import cv2
import os
import sys
import numpy as np
import rknn_yolov8_config as config
from post_process import PostProcess
from rknnlite.api import RKNNLite
from time import time

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from common_utils.logging_utils import LoggingUtilities  # noqa: E402

IMG_SIZE = config.IMG_SIZE
CLASSES = config.CLASSES

# limiting the threads to the big cores, ensures everything runs faster
# the difference gains us about 10 FPS AND uses less CPU
os.environ['OMP_NUM_THREADS'] = '2'


class RKYoloV8:

    def __init__(self, video_path: str, model_path: str):

        self.post_process = PostProcess()
        self.logger = LoggingUtilities.console_out_logger("Main Script")

        # instantiate RKNN API
        self.rknn = RKNNLite()

        self.streaming_object = self.create_streaming_object(video_path)
        self.model = self.load_model(model_path)
        self.logger.info("Model Loaded")

        self.run_time_environment()
        self.logger.info("Run time environment created")
        self.inferencing()

    def load_model(self, model_path):

        model = self.rknn.load_rknn(model_path)
        if model != 0:
            self.logger.debug("Model load error, exiting....")
            sys.exit()

        return model

    def run_time_environment(self):

        status = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

        if status != 0:
            self.logger.debug('Run time instantiation failed, exiting...')
            sys.exit()
        else:
            self.logger.info('RKNN run time environment created')

    def create_streaming_object(self, path: str):

        stream_object = cv2.VideoCapture(path)
        assert stream_object.isOpened(), "Error reading video file"

        return stream_object

    def pre_process(self):

        # read video file or stream
        success, frame = self.streaming_object.read()

        if not success:
            self.logger.info('Data read failure or video over, exiting...')
            self.shutdown()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        shape = frame.shape[:2]

        if shape != IMG_SIZE:
            return self.letter_box(frame, shape, (640, 640), (0, 0, 0))

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

    def inferencing(self):

        frame_counter = 0
        total_inferencing_latency = 0
        total_post_process_latency = 0

        while True:

            frame = self.pre_process()

            # Inference
            start = time()
            output = self.inference(frame)
            end = time()
            inferencing_latency = 1000 * round((end - start), 2)
            total_inferencing_latency += inferencing_latency

            start_process_time = time()
            boxes, classes, scores = self.post(output)
            end_process_time = time()

            post_process_latency = 1000 * round((end_process_time -
                                                 start_process_time), 2)
            total_post_process_latency += post_process_latency

            frame_counter += 1

            if frame_counter % 100 == 0:
                avg_inferencing_latency = round((total_inferencing_latency /
                                                 frame_counter), 2)
                avg_post_latency = round((total_post_process_latency /
                                          frame_counter), 2)
                inference_fps = round((1000 / avg_inferencing_latency), 2)
                overall_fps = round((1000 /
                                     (avg_inferencing_latency +
                                      avg_post_latency)), 2)
                self.logger.info(f'Avg inferencing(ms): {avg_inferencing_latency}, inferencing FPS: {inference_fps}, avg post-process(ms): {avg_post_latency}, overall FPS: {overall_fps}')  # noqa: E501

            # draw boxes on image
            self.draw(frame, boxes, classes, scores)

            key = cv2.waitKey(1)
            if key == ord('q'):
                self.shutdown()

    def inference(self, frame):

        return self.rknn.inference(inputs=[(np.expand_dims(frame, 0))])

    def post(self, data):

        return self.post_process.post_process(data)

    def draw(self, image, boxes, classes, scores):
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(_b) for _b in box]

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
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
