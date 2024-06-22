# Computer Vision Demo Reel

Computer vision demos related to classification, object detection, and segmentation for commercial and industrial use cases. I have a large number of computer vision benchmarking scripts, demos, experiments, solution ideas and the like that aren't necessarily big enough for a stand-alone project but are still worth sharing; the goal of this repo is to aggregate them all into one place for later use. 

**General Approach:** everything shared here should be *"good enough"* to be useful or for the next stage of a conversation to be around what needs to built around the demo to deliver a minimal viable product or proof of concept. E.g., it doesn’t just count cars, it can count cars, handle exceptions, has logic to handle specific events and creates a JSON payload for external systems to consume.

*TL/DR - this repo will lean heavily towards practical examples of how to use these technologies, and will often include things like database integration, slack integration, monitoring dashboards, etc.*

## What we have so far

### YOLOv8 on the Rockchip 3588 NPU 

Using smart thread management and asynchronous functions, we're able to run two video streams simultaneously at about 30 FPS with post processing and ~48 FPS/throughput for inferencing. Not bad for a device that fits in the palm of your hand. **Note:** I've tried this with as many as six streams: three was at mid 20s FPS and 4-5 was around 18-19, six was around 12-14 with a lot of stuttering. I.e., 2-3 streams seems optimal.

![Rockchip RK3588 YOLOv8 NPU](images/rk3588.gif)
* Trying to do screen capture on top of the two streams grinds things to a halt, so recording with my phone was the best option at the moment. 
* Adding the latency to draw boxes + render the frame costs about 2-3 FPS, so the real on screen FPS is closer to 27-28.
* Both videos are 640 x 360


### YOLOv8 People counting and tracking

Counting people as they move past a certain point or "border line" in a video. E.g., people going up or down in an escalator. The entry/exit data, total number of people in the frame, and FPS is collected into a JSON payload for sharing with/transmitting to other systems. 

![People Counting GIF](images/escalator_count.gif)

**Note #1:** FPS refers to processing speed, not the rendering speed which is ~30 FPS for the original video and around 10 FPS for the gif

**Note #2:** in a real life implementation the org would have systems/technology in place that display and store video, so we probably wouldn't render/show video with detections, we would instead just make the data available for later view/analysis whether that's storing the data + the video with detections or just storing the data.

### YOLOv8 Multi-Class Counting

Counting entrances and exits for several different things or classes, think cars going by, people, people on bycycles, dogs, etc. Similar to the above, the demo generates a JSON payload with entry/exit data for each class, and there is an alternate version that transmits data via MQTT to be recorded in InfluxDB for display via Grafana. 

![Multi Class Counting GIF](images/multi_classv4.gif)

**Note #1:** FPS refers to processing speed, not the rendering speed which is ~24 FPS for the original video and around 20 FPS for the gif

**Note #2:** the dashboard updates every 5 seconds vs the on screen data updating with every frame, so the dashboard lags the events in the gif/video.




