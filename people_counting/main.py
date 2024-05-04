# Markham Lee 2023 - 2024
# People counting script via YOLOV8 - uses Ultralytics built
# in class counting functions.
# TODO:
# * Suppress console output with constant inference latency, etc.
# * Calculate real time FPS - show on screen
# * Can we count total detections anywhere in the frame?
# * Create ONNX version for deployment on a small edge device
# * Ideal: version that uses Rockchip 3588 NPU
import sys
import os
import cv2
import json
import torch
from ultralytics import YOLO
from ultralytics.solutions import object_counter


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.logging_utils import logger  # noqa: E401
from utils.com_utilities import CommUtilities


class PeopleCounter:

    def __init__(self, model_name: str, file_path: str):

        # logger
        self.logger = logger

        # Instantiate comms utilities
        self.comms = CommUtilities()

        # get unique ID for sending MQTT messages
        self.UID = self.comms.getClientID()

        # load MQTT env vars & mqtt client
        self.load_mqtt()

        self.TOPIC = "/computer_vision/count_tests"

        # get model
        self.model = self.get_model(model_name)

        # Get streaming object; use width and height from stream object
        # to calculate the image's midpoint to place the line used to
        # count people moving in and out of the frame.
        self.stream_object, width, height \
            = self.get_streaming_object(file_path)

        # get counting object
        self.count_object = self.get_counting_object(width, height)

        # analyze video
        self.analyze_video(self.stream_object)

    def load_mqtt(self):

        # load env vars for MQTT connection

        self.MQTT_USER = os.environ['MQTT_USER']
        self.MQTT_SECRET = os.environ['MQTT_SECRET']
        self.MQTT_BROKER = os.environ["MQTT_BROKER"]
        self.MQTT_PORT = int(os.environ['MQTT_PORT'])
        self.logger.info(self.MQTT_PORT)

        self.mqtt_client, code = self.comms.mqtt_client(self.UID,
                                                        self.MQTT_USER,
                                                        self.MQTT_SECRET,
                                                        self.MQTT_BROKER,
                                                        self.MQTT_PORT)
        self.logger.info("MQTT Client Created")

    def get_model(self, model_name):

        # CUDA acceleration
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        self.logger.info(f'Using device: {device}')

        # create model
        model = YOLO(model_name)
        model.to(device)

        return model

    def get_streaming_object(self, file_path):

        cap = cv2.VideoCapture(file_path)
        assert cap.isOpened(), "Error reading video file"

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                               cv2.CAP_PROP_FRAME_HEIGHT,
                                               cv2.CAP_PROP_FPS))

        self.logger.info(f'Video FPS: {fps}')

        return cap, w, h

    def get_counting_object(self, width: int, height: int):

        line_points = [(width / 2, 0), (width / 2, height)]

        counter = object_counter.ObjectCounter()
        counter.set_args(view_img=True,
                         reg_pts=line_points,
                         classes_names=self.model.names,
                         draw_tracks=False,
                         line_thickness=2)

        return counter

    def analyze_video(self, stream_object: object):

        while stream_object.isOpened():
            success, frame = stream_object.read()
            if not success:
                logger.info('No file or processing complete')
                break

            # set classes to zero because we're only going to count
            # people entering and exiting.
            classes = [0]

            # the video data object contains extensive data on each frame of
            # the video video shape, xy coordintes for each object, object
            # classes and data on inferencing and processing speed.
            video_data = self.model.track(frame, persist=True,
                                          show=False, classes=classes)

            # increment the "counting fields" in and out
            # returned frame could be saved for later viewing
            frame = self.count_object.start_counting(frame, video_data)

            # parse out key data from the video data object
            fps = self.parse_video_data(video_data)

            # Send the MQTT message with the counts, using MQTT as it uses
            # minimal compute power, for a solution that would likely be
            # deployed on a low power edge device.
            self.send_payload(fps)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        stream_object.release()
        cv2.destroyAllWindows()
        self.mqtt_client.loop_stop()

    def parse_video_data(self, data: object) -> dict:

        # json_data = json.loads(data.tojson())
        # extracted_data = json_data

        # count = extracted_data

        self.logger.info(f'the count data is: {data}')

        inferencing_speed = round((sum(data[0].speed.values())), 2)
        return round((1000 / inferencing_speed), 2)

    def send_payload(self, fps):

        payload = {"incoming": int(self.count_object.in_counts),
                   "outgoing": int(self.count_object.out_counts),
                   "fps": float(fps)}

        # convert payload to json for sending out via MQTT
        payload = json.dumps(payload)

        # send data
        self.mqtt_client.publish(self.TOPIC, payload)


count = PeopleCounter("yolov8m", "videos/video.mp4")
count()
