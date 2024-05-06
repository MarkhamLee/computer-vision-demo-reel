# Computer Vision Demo Reel

Computer vision demos related to object classification, detection, and segmentation for commercial and industrial purposes. I have a large number of computer vision benchmarking scripts, demos, experiments, solution ideas and the like that aren't necessarily big enough for a stand-alone project but are still worth sharing; the goal of this repo is to aggregate them all into one place for later use. 

**General Approach:** everything shared here should be *"good enough"* to be useful or for the next stage of conversation to be around what needs to built around the demo to deliver a minimal viable product or proof of concept. E.g., it doesn’t just count cars, it can count cars, handle exceptions, has logic to handle specific events and creates a JSON payload for external systems to consume.

*TL/DR - this repo will lean heavily towards practical examples of how to use these technologies, and will often include things like database integration, slack integration, monitoring dashboards, etc.*

## What we have so far

### YOLOv8 People counting and tracking

Counting people as they move past a certain point or "border line" in a video. E.g., people passing the line from the left are entering an area and people passing the line from the right are leaving it. The generated data payload includes ingress count, egress count, fps and the total people in the frame at a given time.

![People Counting GIF](images/people_counter_detections.gif)


**Note:** in a real life implementation the org would already have video systems in place, so we probably wouldn't render video with detections, we would instead just make the data available.





