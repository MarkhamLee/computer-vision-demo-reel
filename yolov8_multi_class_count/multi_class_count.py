# Markham Lee 2024
# Computer Vision Demo Reel
# https://github.com/MarkhamLee/computer-vision-demo-reel
# Multi-class object counter, the user would provide the classes they want to
# track as a paramter and the script will count ingress/egress vs a line drawn
# in the center of the frame. The script will then generate a JSON payload with
# the ingress/egress count for each class of item and the FPS, this data is
# logged to console once per second.
# TODO:
# Speed this up, the FPS drops off by about 2/3rds after the first few seconds
# however, the GPU utilization is around 20-25% and the CPU is at around 5%,
# this leads me to believe the bottleneck is code related, need to track
# that down.
# Create variants for low powered edge devices, e.g., Orange Pi 5+/Rockchip
# 3588 device, Raspberry Pi 4B or 5.
import cv2
import gc
import os
import sys
import torch
import pandas as pd
from ultralytics import YOLO
from ultralytics.solutions import object_counter


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from common_utils.logging_utils import LoggingUtilities  # noqa: E402

logger = LoggingUtilities.console_out_logger("Multi-class count")


class PeopleCounter:

    def __init__(self, model_name: str, file_path: str, classes: list):

        # logger
        self.logger = logger

        # get model
        self.model = self.get_model(model_name)

        self.total_count = 10

        # Get streaming object; use width and height from stream object
        # to calculate the image's midpoint to place the line used to
        # count people moving in and out of the frame.
        self.stream_object, width, height, orig_fps \
            = self.get_streaming_object(file_path)

        # get counting object
        self.count_object = self.get_counting_object(width, height)

        # analyze video
        self.analyze_video(self.stream_object, orig_fps, classes)

    def get_model(self, model_name: str) -> object:

        # CUDA acceleration
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        self.logger.info(f'Using device: {device}')

        # create model
        model = YOLO(model_name)
        model.to(device)

        return model

    def get_streaming_object(self, file_path: str):

        cap = cv2.VideoCapture(file_path)
        assert cap.isOpened(), "Error reading video file"

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                               cv2.CAP_PROP_FRAME_HEIGHT,
                                               cv2.CAP_PROP_FPS))

        self.logger.info(f'Original video FPS: {fps}')

        return cap, w, h, fps

    def get_counting_object(self, width: int, height: int):

        line_points = [(width / 2, 0), (width / 2, height)]

        counter = object_counter.ObjectCounter()
        # set "view_img" to false if you don't want to view the video
        counter.set_args(view_img=True,
                         reg_pts=line_points,
                         classes_names=self.model.names,
                         draw_tracks=False,
                         line_thickness=2)

        return counter

    def analyze_video(self, stream_object: object, orig_fps: float,
                      classes: list):

        count = 0

        while stream_object.isOpened():
            success, frame = stream_object.read()
            if not success:
                logger.info('No file or processing complete')
                break

            # the video data object contains extensive data on each frame of
            # the video video shape, xy coordintes for each object, object
            # classes and data on inferencing and processing speed.
            video_data = self.model.track(frame, persist=True,
                                          show=False, verbose=False,
                                          classes=classes)

            # parse out key data from the video data object
            fps = self.parse_video_data(video_data)

            # Add FPS to the frame
            cv2.putText(frame, f"FPS: {fps}",
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # increment the "counting fields" in and out
            # returned frame can be saved for later viewing
            frame = self.count_object.start_counting(frame, video_data)

            count += 1

            # pull dictionary with in/out counts for each class from the
            # counting object
            nested_payload = self.count_object.class_wise_count

            # use pandas to flatten nested dictionary
            intermediate_df = pd.json_normalize(nested_payload, sep='_')
            payload = intermediate_df.to_dict(orient='records')[0]

            # add FPS data to the base payload
            payload.update({"FPS": fps})

            # Printing out the JSON payload, once per second
            # Use the video's original FPS as a way to time
            # a second based on having received x # of frames.
            if count == orig_fps:
                self.logger.info(f'Current payload: {payload}')
                count = 0

            # garbage collection
            del frame, fps, payload, intermediate_df
            gc.collect()

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        stream_object.release()
        cv2.destroyAllWindows()

    def parse_video_data(self, data: object) -> dict:

        inferencing_speed = round((sum(data[0].speed.values())), 2)
        return round((1000 / inferencing_speed), 2)


# pass the model name, path to video and list of classes to be tracked
count = PeopleCounter("yolov8m", "../videos/videos2.mp4", [0, 1, 2])
count()
