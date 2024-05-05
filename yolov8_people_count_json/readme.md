## YOLO V8 ingress/egress counter with JSON output  

Sample implementation of YOLO V8's count and tracker features that allow you to count people as they pass through a pre-defined line in the video frame, approaching from the right of the line counts as an ingress and approaching the line from the left counts as an egress. The example also counts the # of people in the frame at a given time. All of the data (ingress, egress, people present and fps) is aggregated into a JSON payload, which could be used to write to a data base, generate MQTT messages, etc. The general idea was to build something that goes beyond the usual "counting" or "detection" demos by adding the ability to collect data that other systems could use.

