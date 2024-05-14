## YOLO V8 ingress/egress counter with JSON output  

Sample implementation of YOLO V8's count and tracker features that allow you to count people as they pass through a pre-defined line in the video frame, in this case we're counting people going up or down an escalator. The example also counts the # of people in the frame at a given time. This could be used to count people entering or exiting an area, identify busy times, etc.
All of the data (ingress, egress, people present and fps) is aggregated into a JSON payload, which could be used to write to a database, generate MQTT messages, etc. The general idea was to build something that goes beyond the usual "counting" or "detection" demos by adding the ability to collect data that other systems could use.

![People Counting GIF](../images/escalator_count.gif)

**Note 1:** FPS refers to processing speed, not the rendering speed which is ~30 FPS for the original video and around 10 FPS for the gif

**Note #2:** in a real life implementation the org would have systems/technology in place that display and store video, so we probably wouldn't render/show video with detections, we would instead just make the data available for later view/analysis whether that's storing the data + the video with detections or just storing the data.