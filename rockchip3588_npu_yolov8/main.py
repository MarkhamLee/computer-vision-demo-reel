# Based on/inspired by: https://github.com/MarkhamLee/rknn_model_zoo/blob/main/examples/yolov8/python/yolov8.py
# YOLOv8 object counter running on a Rockchip3588 NPU via the
# RKNN process. For YOLO models the RKNN conversion process pushes
# some of the post processing steps that would normally occur on a GPU
# to the CPU: https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.md
# the extra steps on CPU take nearly as long as the inferencing.
# Temps barely move about 43C for running inferencing w/o post process,
# adding post process pushes temps to 55-58C and maxes out all CPUs, 
# as opposed to sitting around 5-10% otherwise. 
# TODO: optimize post process time, will probably use cython and/or look
# into the longshot of possibly moving those to the device's GPU. That being said
# this probably just needs to be implemented in C++.
# Next Steps: add JSON output of data from video, add unique tracking and counting
import cv2
import gc
import os
import sys
import numpy as np
import rknn_yolov8_config as config
from post_process import PostProcess
from rknnlite.api import RKNNLite
from statistics import mean
from time import time

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from common_utils.logging_utils import LoggingUtilities  # noqa: E402

IMG_SIZE = config.IMG_SIZE
CLASSES = config.CLASSES


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

    def create_streaming_object(self, path: str):

        stream_object = cv2.VideoCapture(path)
        assert stream_object.isOpened(), "Error reading video file"

        w, h, fps = (int(stream_object.get(x))
                     for x in (cv2.CAP_PROP_FRAME_WIDTH,
                               cv2.CAP_PROP_FRAME_HEIGHT,
                               cv2.CAP_PROP_FPS))

        self.logger.info(f'Video loaded, original FPS: {fps}, width: {w}, height: {h}')

        return stream_object

    def run_time_environment(self):

        status = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)

        if status != 0:
            self.logger.debug('Run time instantiation failed, exiting...')
            sys.exit()
        else:
            self.logger.info('RKNN run time environment created')

    def inferencing(self):

        counter = 0
        latency_list = []
        process_list = []

        while True:

            # read video file or stream
            success, frame = self.streaming_object.read()

            if not success:
                self.logger.info('Data read failure or video over, exiting...')
                self.shutdown()

            # instead of resizing the image, we letter box it so it
            # "fits" the 640 x 640 shape the model requires 
            frame = self.letter_box(frame, new_shape=(IMG_SIZE[1],
                                                      IMG_SIZE[0]), pad_color=(0, 0, 0))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
             # make a copy for display as we can't use the expanded version below
             # for displaying via OpenCV
             # also, draw the boxes on the SAME image used for inferencing, otherwise
             # there is a good chance some of your boxes won't be on the objects in
             # more densely packed videos.
            image = frame.copy()
            frame = np.expand_dims(frame, 0) # model expects (1, 3, 640, 640)

            # Inference
            start = time()
            outputs = self.rknn.inference(inputs=[frame])  # noqa: F841
            end = time()
            inferencing_latency = 1000 * round((end - start), 2)
            counter += 1

            latency_list.append(inferencing_latency)
           
            start_process_time = time()
            boxes, classes, scores = self.post_process.post_process(outputs)
            end_process_time = time()

            post_process_latency = 1000 * round((end_process_time -
                                                 start_process_time), 2)

            process_list.append(post_process_latency)
        
            if counter == 10:
                counter = 0
                avg_latency = round((mean(latency_list)), 2)
                avg_post_process_latency = round((mean(process_list)), 2)
                avg_total_latency = avg_latency + avg_post_process_latency
                inferencing_fps = round((1000/avg_latency), 2)
                self.logger.info(f'Average latency: {avg_latency}, Avg. FPS: {inferencing_fps}, post process latency: {avg_post_process_latency}')  # noqa: E501

            # draw boxes on image
            # given this would be deployed at the edge, we only need to draw boxes on
            # a screen for testing, edge deployment would be more about data collection
            self.draw(image, boxes, classes, scores)

            # can also add ability to save the frames here/create a video
            # with the detections.

            key = cv2.waitKey(1)
            if key == ord('q'):
                self.shutdown()

    def letter_box(self, im, new_shape, pad_color):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=pad_color)  # add border

        return im

    def draw(self, image, boxes, classes, scores):
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(_b) for _b in box]
            
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
            cv2.imshow("full post process result", image)

    def shutdown(self):

        self.rknn.release()
        cv2.destroyAllWindows()
        self.streaming_object.release()
        sys.exit()


a = RKYoloV8("../videos/cars.mp4",
             "../rknn_models/yolov8n.rknn")
