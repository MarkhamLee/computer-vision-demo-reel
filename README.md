# Computer Vision Demo Reel

Computer vision demos related to classification, object detection, and segmentation for commercial and industrial use cases. I have a large number of computer vision benchmarking scripts, demos, experiments, solution ideas and the like that aren't necessarily big enough for a stand-alone project but are still worth sharing; the goal of this repo is to aggregate them all into one place for later use. 

**General Approach:** everything shared here should be *"good enough"* to be useful or for the next stage of conversation to be around what needs to built around the demo to deliver a minimal viable product or proof of concept. E.g., it doesn’t just count cars, it can count cars, handle exceptions, has logic to handle specific events and creates a JSON payload for external systems to consume.

*TL/DR - this repo will lean heavily towards practical examples of how to use these technologies, and will often include things like database integration, slack integration, monitoring dashboards, etc.*

## What we have so far

### YOLOv8 People counting and tracking

Counting people as they move past a certain point or "border line" in a video. E.g., people passing the line from the left are entering an area and people passing the line from the right are leaving it. The entry/exit data, total number of people in the frame, and FPS is collected into a JSON payload for sharing/transmitting with other systems. 

![People Counting GIF](images/people_counter_detections_v2.gif)

**Note #1:** FPS refers to processing speed, not the rendering speed which is ~30 FPS for the original video and around 10 FPS for the gif

**Note #1:** the line in the center is generated automatically by calculating the midpoint of the image and then drawing the line on each frame. In an actual implementation the line could be moved to where it best fits the use case(s).

**Note #2:** in a real life implementation the org would already systems/technology in place that display and store video, so we probably wouldn't render/show video with detections, we would instead just make the data available for later view/analysis.

### YOLOv8 Multi-Class Counting

Counting entrances and exits for several different things or classes, think cars going by, people, people on bycycles, dogs, etc. Similar to the above, the demo generates a JSON payload with entry/exit data for each class, and there is an alternate version that transmits data via MQTT to be recorded in InfluxDB for display via Grafana. 

![Multi Class Counting GIF](images/multi_count_dashboard_v2.gif)

**Note #1:** FPS refers to processing speed, not the rendering speed which is ~30 FPS for the original video and around 10 FPS for the gif

**Note #2:** the dashboard updates every 5 seconds vs the on screen data updating with every frame, so the dashboard lags the data on the gif/video.

**Note #3:** similar to the above, there would probably not be a need to render the video in real time, we would likely just capture the data and transmit it to where it needs to go.



