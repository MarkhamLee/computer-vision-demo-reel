# Markham Lee 2024
# Computer Vision Demo Reel
# https://github.com/MarkhamLee/computer-vision-demo-reel
# Demos a full video analytics pipeline for the Multi-class object counter by
# adding the ability to transmit the JSON payload via MQTT. Usage: the user
# would provide the classes they want to track as a paramwter, plus configure
# environmental variables for connecting to a MQTT broker and provide a path
# to a video file. From there the script will count ingress/egress for each
# detected class vs a line drawn in the center of the frame, calculate FPS,
# create a JSON payload and the send the data to the MQTT broker.
# Note: if framerates on Windows drop significantly after a few seconds,
# updating your Ultralytics library shouild resolve it.
# TODO:
# create variants for low powered edge devices, e.g., Orange Pi 5+/Rockchip
# 3588 device, Raspberry Pi 4B or 5.
import asyncio
import cv2
import json
import os
import sys
import torch
import pandas as pd
from ultralytics import YOLO
from ultralytics.solutions import object_counter


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from common_utils.logging_utils import LoggingUtilities  # noqa: E402
from common_utils.com_utilities import CommUtilities  # noqa: E402


class PeopleCounter:

    def __init__(self, model_name: str, file_path: str, classes: list):

        # logger
        self.logger = LoggingUtilities.\
            console_out_logger("multi-class count w/ MQTT")

        # Instantiate comms utilities
        self.comms = CommUtilities()

        # get unique ID for sending MQTT messages
        self.UID = self.comms.getClientID()

        # load MQTT env vars & mqtt client
        self.load_mqtt()

        self.TOPIC = "/computer_vision/multi_count"

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
        asyncio.run(self.analyze_video(classes))

    def load_mqtt(self):

        # load env vars for MQTT connection

        self.MQTT_USER = os.environ['MQTT_USER']
        self.MQTT_SECRET = os.environ['MQTT_SECRET']
        self.MQTT_BROKER = os.environ["MQTT_BROKER"]
        self.MQTT_PORT = int(os.environ['MQTT_PORT'])

        self.mqtt_client, code = self.comms.mqtt_client(self.UID,
                                                        self.MQTT_USER,
                                                        self.MQTT_SECRET,
                                                        self.MQTT_BROKER,
                                                        self.MQTT_PORT)
        self.logger.info("MQTT Client Created")

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
        counter.set_args(view_img=False,
                         reg_pts=line_points,
                         classes_names=self.model.names,
                         draw_tracks=False,
                         line_thickness=2,
                         view_out_counts=False,
                         view_in_counts=False)

        return counter

    def pre_process(self):

        success, frame = self.stream_object.read()

        if not success:
            self.logger.info('No file or processing complete')
            self.shutdown()

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    async def analyze_video(self, classes: list):

        frame_count = 0
        fps_sum = 0
        avg_fps = 0
        total_latency = 0

        while True:

            frame = self.pre_process()

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

            total_latency = total_latency + inferencing_latency
            avg_latency = round((total_latency / frame_count), 2)

            # Add FPS to the frame
            cv2.putText(frame, f"AVG FPS: {avg_fps}",
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # increment the "counting fields" in and out
            # returned frame can be saved for later viewing
            frame = self.count_object.start_counting(frame, video_data)

            # send out MQTT message
            # use an async function so the MQTT latency doesn't slow
            # things down.
            payload = await self.send_payload(fps, avg_fps, avg_latency)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("monitor detections", frame)
            key = cv2.waitKey(15)  # add extra delay otherwise it runs too fast

            if key == ord('q'):
                self.shutdown()

            if frame_count % 100 == 0:
                self.logger.info(f'Video data: {payload}')

    def parse_video_data(self, data: object) -> dict:

        inferencing_latency = round((sum(data[0].speed.values())), 2)
        return round((1000 / inferencing_latency), 2), inferencing_latency

    async def send_payload(self, fps, avg_fps, avg_latency):

        # pull dictionary with in/out counts for each class from the
        # counting object
        nested_payload = self.count_object.class_wise_count

        # use pandas to flatten nested dictionary received from above
        intermediate_df = pd.json_normalize(nested_payload, sep='_')
        payload = intermediate_df.to_dict(orient='records')[0]

        # add FPS & latency data to the base payload
        payload.update({"FPS": fps,
                        "Avg_FPS": avg_fps,
                        "avg_latency": avg_latency})

        # convert payload to json for sending out via MQTT
        payload = json.dumps(payload)

        # send data
        self.mqtt_client.publish(self.TOPIC, payload)

        return payload

    def shutdown(self):
        self.stream_object.release()
        cv2.destroyAllWindows()
        sys.exit()


# pass the model name, path to video and list of classes to be tracked
count = PeopleCounter("yolov8n", "../videos/multi_class_video.mp4", [0, 1, 2])
