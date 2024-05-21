import cv2
import gc
import json
import os
import sys
import torch
from ultralytics import YOLO

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from common_utils.logging_utils import LoggingUtilities  # noqa: E402

logger = LoggingUtilities.console_out_logger("NCNN Tracker")

torch.set_num_threads = 1


class ObjectTracker:

    def __init__(self, model_name: str, file_path: str):

        self.logger = logger

        self.model = self.get_model(model_name)

        self.stream_object, width, height, orig_fps \
            = self.get_streaming_object(file_path)

        # analyze video
        self.analyze_video(self.stream_object,
                           orig_fps)

    def get_model(self, model_name: str) -> object:

        # create model
        model = YOLO(model_name)

        return model

    def get_streaming_object(self, file_path: str):

        cap = cv2.VideoCapture(file_path)
        assert cap.isOpened(), "Error reading video file"

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                               cv2.CAP_PROP_FRAME_HEIGHT,
                                               cv2.CAP_PROP_FPS))

        self.logger.info(f'Original video FPS: {fps}')

        return cap, w, h, fps

    def analyze_video(self, stream_object: object, orig_fps: float):

        count = 0
        fps_sum = 0
        frame_count = 0
        avg_fps = 0
        total_latency = 0

        while stream_object.isOpened():
            success, frame = stream_object.read()
            if not success:
                logger.info('No file or processing complete')
                logger.info(f'Video Complete, average FPS: {avg_fps}')
                self.shutdown()
                break

            # the video data object contains extensive data on each frame of
            # the video video shape, xy coordintes for each object, object
            # classes and data on inferencing and processing speed.
            video_data = self.model.track(frame, persist=True,
                                          verbose=False, show=False)
            count += 1

            # parse out key data from the video data object
            fps, inferencing_latency = self.parse_video_data(video_data)

            frame_count += 1
            fps_sum = fps_sum + fps
            avg_fps = round((fps_sum/frame_count), 2)

            total_latency += inferencing_latency

            avg_latency = round((total_latency/frame_count), 2)

            frame_message = (f'Avg FPS: {avg_fps}')
            self.write_on_frame(frame, frame_message, (5, 25))

            # create JSON payload
            payload = self.build_payload(avg_fps, avg_latency)

            if count == orig_fps:
                self.logger.info(f'Current payload: {payload}')
                count = 0

            del payload, avg_latency
            gc.collect()

            # Visualize the results on the frame - I only use
            # annotation and showing the video to "eyeball validate"
            # that things are operating properly and then I comment it out
            annotated_frame = video_data[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.shutdown()
                break

    def write_on_frame(self, frame: object, text: str,
                       coordinates: tuple):

        cv2.putText(frame, text, coordinates,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def parse_video_data(self, data: object) -> dict:

        inferencing_latency = round((sum(data[0].speed.values())), 2)
        return round((1000 / inferencing_latency), 2), inferencing_latency

    def build_payload(self, avg_fps: float, inferencing_latency) -> dict:

        # note: we force int as the system will sometimes count a partial cross
        # as a fraction of a count until they've completely crossed the line.
        payload = {"avg_fps": avg_fps,
                   "avg_inferencing_latency(ms)": inferencing_latency}

        # convert to json
        return json.dumps(payload)

    def shutdown(self):

        self.stream_object.release()
        cv2.destroyAllWindows()


# pass the model name, path to video and list of classes to be tracked
tracking = ObjectTracker("yolov8n.pt",
                         "../videos/rural_highway.mp4")
