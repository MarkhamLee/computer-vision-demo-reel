# Computer Vision Demo Reel

A collection of real-time video analysis pipelines, each intended as a proof of concept or foundation for a more complete video analytics solution. These demos produce JSON outputs for integration into existing systems, they're not demos that are thrown away after a meeting, they're built to be tested against real use cases. 

## What we have so far

### YOLOv8 on the Rockchip 3588 NPU 

A YOLOv8 inference pipeline optimized for the Rockchip RK3588 system-on-chip, 
using asynchronous processing and careful thread management to achieve 20+ FPS on 2-4 360p video streams on a device that is roughly the size of an average Smart Phone. 

**Key architectural decision:** limiting the pipeline to only use two threads forces data loading, video rendering and post processing to work on the machine's high performance cores, and negates the performance penalty of using and big and little cores together. Async processing and routing each video stream to an individual NPU core speeds up inferencing, while allowing multiple streams to more effectively share inferencing cores. Utilization wise this meant going from the device struggling to reach 20 FPS with one stream with all eight cores at over 90% utilization, to being able to support multiple streams at over 25% with only two of the cores over 90%. The reduced CPU utilization freed up system resources for sending and receiving data, and other tasks required for the device to be part of a broader data ingestion pipeline. 

The async threading pattern developed here was later adapted into a production-grade multi-model pipeline at Fluffy Pet Technologies, a more fully async and multi-threaded implementation running detection, tracking, pose estimation, and VLM-based analysis at 40+ FPS on 4K video on NVIDIA GPUs.

![Rockchip RK3588 YOLOv8 NPU](images/rk3588.gif)
* Trying to do screen capture on top of the two streams grinds things to a halt, so recording with my phone was the best option at the moment. 
* Adding the latency to draw boxes + render the frame costs about 2-3 FPS, so the real on screen FPS is closer to 27-28.
* Both videos are 640 x 360

**Note 1:** this was built before Ultralytics added native RKNN export support; meaning: many of the difficulties and work arounds documented in this demo's sub-folder no longer apply, but the broader architecture still does. 


### YOLOv8 People counting and tracking

Counts objects crossing a defined boundary line in a video frame, split by direction of travel — useful for people or vehicle flow monitoring, entrance/exit counting, or occupancy tracking. Produces a JSON payload per frame containing 
ingress count, egress count, current objects in frame, and processing FPS, suitable for writing to a time-series database like InfluxDB. 

![People Counting GIF](images/escalator_count.gif)

**Note #1:** FPS refers to processing speed, not the rendering speed which is ~30 FPS for the original video and around 10 FPS for the gif

**Note #2:** in a real life implementation the org would have systems/technology in place that display and store video, so we probably wouldn't render/show video with detections, we would instead just make the data available for later view/analysis whether that's storing the data + the video with detections or just storing the data.

### YOLOv8 Multi-Class Counting

Extends the single-class counter to simultaneous multi-class tracking — people, cars, and bicycles counted independently in real time. Built around a city traffic monitoring scenario; data is pushed via MQTT through a Node-RED pipeline into 
InfluxDB for display on a Grafana dashboard, demonstrating near-real-time traffic analytics from edge to visualization.

![Multi Class Counting GIF](images/multi_classv4.gif)

**Note #1:** FPS refers to processing speed, not the rendering speed which is ~24 FPS for the original video and around 20 FPS for the gif

**Note #2:** the dashboard updates every 5 seconds vs the on screen data updating with every frame, so the dashboard lags the events in the gif/video.




