## Object Tracking with YOLO v8 & NCNN on an Orange Pi 5+ 16 GB (Rockchip 3588) - WIP

Work in progress/experimenting with optimization/acceleration formats for low power devices.

#### Conversion Steps

* Convert to NCNN format via this command

~~~
yolo export model=yolo8n.pt format=ncnn int8=true
~~~

* Make note of pointing to the entire NCNN directory and not just a file 
* Make sure the file is 640 x 360 or adjust the image format via the "imgsz" paramter

I was able to get about 16 FPS if I was rendering the video and 18 if I was just collecting data to build JSON data paylaods, while using the video marked "rural_highway" (original FPS of 23) in the videos folder. With a device like this, you would probably be using it at the edge to gather data rather than view videos with detections so I feel the later number is more useful. 

### TODO
* There is an error message saying that that YOLO can't determine the task and it's assuming "task=detect", but tracking seems to be working and the unique IDs are being assigned to each object despite "technically" invoking the model wrong. Task 1 is to make sure the tracking is occuring properly
* Once the above works I'll either add counting or some other type of analysis, big thing now is just getting NCNN working properly on the device.
