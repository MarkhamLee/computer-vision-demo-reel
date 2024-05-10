# Markham Lee 2024
# Computer Vision Demo Reel
# https://github.com/MarkhamLee/computer-vision-demo-reel
# Counting people crossing a line in a video file I.e., ingress and egress
# from an area. Analyzes a video file and generates a JSON payload with
# the ingress and egress count, FPS for the last frame and total people
# in the space at a given time
# Note: if framerates on Windows drop significantly after a few seconds,
# updating your Ultralytics library shouild resolve it.
# TODO:
# Create variants for low powered edge devices
# Create variant using Rockchip NPU. Add parameter for
# passing room capacity to calculate % full.
import cv2
import os
import json
import sys
import torch
from ultralytics import YOLO
from ultralytics.solutions import object_counter


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from common_utils.logging_utils import LoggingUtilities  # noqa: E402


class PeopleCounter:

    def __init__(self, model_name: str, file_path: str):

        # logger
        self.logger = LoggingUtilities.console_out_logger("people count")

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
        self.analyze_video(self.stream_object, orig_fps)

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
        # set "view_img" to false if you don't want to view the video
        counter.set_args(view_img=True,
                         reg_pts=line_points,
                         classes_names=self.model.names,
                         draw_tracks=False,
                         line_thickness=2,
                         view_out_counts=False,
                         view_in_counts=False)

        return counter

    def analyze_video(self, stream_object: object, orig_fps: float):

        frame_count = 0
        fps_sum = 0
        count = 0
        avg_fps = 0

        while stream_object.isOpened():
            success, frame = stream_object.read()
            if not success:
                self.logger.info('No file or processing complete')
                self.logger.info(f'Video Complete, average FPS: {avg_fps}')
                break

            # set class index to zero as that's the index for people
            # and we're only counting people.
            # TODO: provide class as a parameter
            classes = [0]

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

            # TODO: update so that it can be used to count
            # instances of all classes, not just people
            # i.e., make this generic
            people_count = len((video_data[0]).boxes)

            # Add FPS and people count(s) to the frame
            cv2.putText(frame, f"Avg. FPS: {avg_fps}",
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Total people present: {people_count}",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Incoming Persons: {int(self.count_object.in_counts)}",  # noqa: E501
                        ((30), 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Outgoing Persons: {int(self.count_object.out_counts)}",  # noqa: E501
                        (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # increment the "counting fields" in and out
            # returned frame could be saved for later viewing
            frame = self.count_object.start_counting(frame, video_data)

            count += 1

            # create a JSON payload for consumption by other
            # systems.
            payload = self.build_payload(fps, people_count,
                                         avg_fps, inferencing_latency)

            if count == orig_fps:
                self.logger.info(f'Current payload: {payload}')
                count = 0

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        stream_object.release()
        cv2.destroyAllWindows()

    def parse_video_data(self, data: object) -> dict:

        inferencing_latency = round((sum(data[0].speed.values())), 2)
        return round((1000 / inferencing_latency), 2), inferencing_latency

    def build_payload(self, fps: float, count: int,
                      avg_fps: float, inferencing_latency) -> dict:

        # note: we force int as the system will sometimes count a partial cross
        # as a fraction of a count until they've completely crossed the line.
        payload = {"total_incoming": int(self.count_object.in_counts),
                   "total_outgoing": int(self.count_object.out_counts),
                   "total_persons_in_view": count,
                   "current_fps": float(fps),
                   "avg_fps": avg_fps,
                   "inferencing_latency(ms)": inferencing_latency}

        # convert to json
        return json.dumps(payload)


count = PeopleCounter("yolov8n", "../videos/videos.mp4")
