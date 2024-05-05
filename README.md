## Computer Vision Demo Reel

Computer vision demos related to object detection, object segmentation and other types of computer vision demos, examples and tutorials for general commercial or industrial purposes. 
I have a large number of computer vision benchmarking scripts, demos, experiments, solution ideas and the like that aren't necessarily big enough for a stand-alone project but are still worth sharing; the goal of this repo is to aggregate them all into one place for later use.

**General Approach:** everything shared here should be *"good enough"* to be useful, e.g., instead of a basic object detection demo, it should be object detection + build a json payload so the data can be exported to other systems; instead of a benchmarking script that estimates performance on a handful of images, it should use a video that's several minutes long and/or provide the user with the ability to use the same type of data they would use if the model was deployed to production.

TL/DR - this will lean heavily towards practical examples of how to use these technologies. 

### What we have so far

#### YOLO V8 People counting and tracking

* **yolov8_people_count_json:** Tracking and counting people as they move into or out of a particular area, using one of YOLO v8s built in tracking functions, which assigns a unique ID to each person in the video and counts people as they move into or out of a designated area. E.g., you could use this to count foot traffic into and out of a retail area. The generated data payload includes ingress count, egress count, fps and total people in the frame at a given time. 





