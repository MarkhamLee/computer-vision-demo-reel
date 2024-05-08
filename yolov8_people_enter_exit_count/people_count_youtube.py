# Markham Lee 2024
# Computer Vision Demo Reel
# https://github.com/MarkhamLee/computer-vision-demo-reel
# Counting people crossing a line in a video file I.e., ingress and egress
# from an area. Analyzes a video file and generates a JSON payload with
# the ingress and egress count, FPS for the last frame and total people
# in the space at a given time
# TODO:
# Modify with the library for creating a cap for YouTube files
# Add average FPS calculation to this one and the
# create variants for low powered edge devices, e.g., ONNX, TF-Lite
# and/or NCNN. Create variant using Rockchip NPU. Add parameter for
# passing room capacity to calculate % full.
import sys
import os
import cv2
import json
import pafy
import torch
from pprint import pprint
from ultralytics import YOLO
from ultralytics.solutions import object_counter


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from common_utils.logging_utils import LoggingUtilities  # noqa: E402

logger = LoggingUtilities.console_out_logger('people_count_youtube')


class PeopleCounter:

    def __init__(self, model_name: str, file_path: str):

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

    def get_streaming_object(self, path: str):

        # extra step for YouTube videos
        youtube_stream = pafy.new(path)

        stream = youtube_stream.getbestvideo(preftype="mp4")

        self.logger.info(f'Incoming stream at {stream.quality}')
        self.logger.info(f'Stream title {stream.title}')

        cap = cv2.VideoCapture(stream.url_https)

        assert cap.isOpened(), "Error reading video file"

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                               cv2.CAP_PROP_FRAME_HEIGHT,
                                               cv2.CAP_PROP_FPS))

        self.logger.info(f'Incoming video FPS: {fps}')

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

    def analyze_video(self, stream_object: object, orig_fps: float):

        count = 0

        while stream_object.isOpened():
            success, frame = stream_object.read()
            if not success:
                logger.info('No file or processing complete')
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
            fps = self.parse_video_data(video_data)

            # TODO: update so that it can be used to count
            # instances of all classes, not just people
            # i.e., make this generic
            people_count = len((video_data[0]).boxes)

            # Add FPS and people count to the frame
            cv2.putText(frame, f"FPS: {fps}",
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Total people present: {people_count}",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # increment the "counting fields" in and out
            # returned frame could be saved for later viewing
            frame = self.count_object.start_counting(frame, video_data)

            count += 1

            # create payload - this data can now be written to a
            # DB, sent out via MQTT or something like AWS SNS
            # for consumption by a data ingestion system.
            payload = self.build_payload(fps, people_count)

            if count == orig_fps:
                pprint(payload)
                count = 0

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        stream_object.release()
        cv2.destroyAllWindows()

    def parse_video_data(self, data: object) -> dict:

        inferencing_speed = round((sum(data[0].speed.values())), 2)
        return round((1000 / inferencing_speed), 2)

    def build_payload(self, fps: float, count: int) -> dict:

        payload = {"total_incoming": int(self.count_object.in_counts),
                   "total_outgoing": int(self.count_object.out_counts),
                   "total_persons_in_view": count,
                   "current_fps": float(fps)}

        # convert to json
        return json.dumps(payload)


count = PeopleCounter("yolov8s", "https://www.youtube.com/watch?v=lnGCrqFq-bc")
count()
