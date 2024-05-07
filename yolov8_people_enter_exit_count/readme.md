## YOLO V8 ingress/egress counter with JSON output  

Sample implementation of YOLO V8's count and tracker features that allow you to count people as they pass through a pre-defined line in the video frame, approaching from the right of the line counts as an ingress and approaching the line from the left counts as an egress. The example also counts the # of people in the frame at a given time. All of the data (ingress, egress, people present and fps) is aggregated into a JSON payload, which could be used to write to a database, generate MQTT messages, etc. The general idea was to build something that goes beyond the usual "counting" or "detection" demos by adding the ability to collect data that other systems could use.

![People Counting GIF](../images/people_counter_detections_v2.gif)

**Note 1:** FPS refers to processing speed, not the rendering speed which is ~30 FPS for the original video and around 10 FPS for the gif

**Note #1:** the line in the center is generated automatically by calculating the midpoint of the image and then drawing the line on each frame. In an actual implementation the line could be moved to where it best fits the use case(s).

**Note #2:** in a real life implementation the org would already systems/technology in place that display and store video, so we probably wouldn't render/show video with detections, we would instead just make the data available for later view/analysis.