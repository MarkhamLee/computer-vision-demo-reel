# Markham Lee 2024
# Computer Vision Demo Reel
# https://github.com/MarkhamLee/computer-vision-demo-reel
# Multi-class object counter, the user would provide the classes they want to
# track as a paramter and the script will count ingress/egress vs a line drawn
# in the center of the frame. The script will then generate a JSON payload with
# the ingress/egress count for each class of item and the FPS, this data is
# logged to console once per second.
# Note: if framerates on Windows drop significantly after a few seconds,
# updating your Ultralytics library shouild resolve it.
# TODO:
# Create variants for low powered edge devices, e.g., Orange Pi 5+/Rockchip
# 3588 device, Raspberry Pi 4B or 5.
import cv2
import os
import sys
import torch
import pandas as pd
from ultralytics import YOLO
from ultralytics.solutions import object_counter


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from common_utils.logging_utils import LoggingUtilities  # noqa: E402


class PeopleCounter():

    def __init__(self, model_name: str, file_path: str, classes: list):

        # logger
        self.logger = LoggingUtilities.console_out_logger("Multi-class count")

        # get model
        self.model = self.get_model(model_name)

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

        # create model
        model = YOLO(model_name)

        # using a more complex check because using .to(device) when
        # its CPU will cause model to not run when using formats like
        # onnx or NCNN.
        if torch.cuda.is_available():
            device = 'cuda:0'
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            model.to(device)

        else:
            device = 'cpu'

        self.logger.info(f'Using device: {device}')

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
        counter.set_args(view_img=False,
                         reg_pts=line_points,
                         classes_names=self.model.names,
                         draw_tracks=False,
                         line_thickness=2)

        return counter

    def pre_process(self, stream_object):

        success, frame = stream_object.read()

        if not success:
            self.logger.info('No file or processing complete')
            self.shutdown()

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def analyze_video(self, stream_object: object, orig_fps: float,
                      classes: list):

        frame_count = 0
        fps_sum = 0
        avg_fps = 0

        while True:

            frame = self.pre_process(stream_object)

            # the video data object contains extensive data on each frame of
            # the video video shape, xy coordintes for each object, object
            # classes and data on inferencing and processing speed.
            video_data = self.model.track(frame, persist=True,
                                          show=False, verbose=False,
                                          classes=classes)

            # parse out key data from the video data object
            fps, inferencing_latency = self.parse_video_data(video_data)

            fps_sum = fps_sum + fps
            frame_count += 1
            avg_fps = round((fps_sum / frame_count), 2)

            # Add FPS to the frame
            cv2.putText(frame, f"Avg FPS: {avg_fps}",
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # increment the "counting fields" in and out
            # returned frame can be saved for later viewing
            frame = self.count_object.start_counting(frame, video_data)

            payload = self.build_payload(fps, avg_fps, inferencing_latency)

            if frame_count % 100 == 0:
                self.logger.info(f'Current payload: {payload}')

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("monitor detections", frame)

            key = cv2.waitKey(10)
            if key == ord('q'):
                self.shutdown()

    def build_payload(self, fps, avg_fps, inferencing_latency):

        # pull dictionary with in/out counts for each class from the
        # counting object
        nested_payload = self.count_object.class_wise_count

        # use pandas to flatten nested dictionary
        intermediate_df = pd.json_normalize(nested_payload, sep='_')
        payload = intermediate_df.to_dict(orient='records')[0]

        # add FPS and latency data to the base payload
        payload.update({"FPS": fps,
                        "Avg_FPS": avg_fps,
                        "inferencing_latency": inferencing_latency})

        return payload

    def parse_video_data(self, data: object) -> dict:

        inferencing_latency = round((sum(data[0].speed.values())), 2)
        return round((1000 / inferencing_latency), 2), inferencing_latency

    def shutdown(self):

        self.stream_object.release()
        cv2.destroyAllWindows()
        self.logger.info("Stream object released, exiting...")
        sys.exit()


# pass the model name, path to video and list of classes to be tracked
count = PeopleCounter("yolov8n.pt",
                      "../videos/multi_class_video.mp4", [0, 1, 2])
