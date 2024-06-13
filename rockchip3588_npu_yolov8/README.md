## YOLOv8 Object Detection on the NPU of a Rockchip 3588 System on Chip (SOC)

An example of how to use asynchronous functions and smart threading management to not only run YOLOv8n on a Rockchip3588 NPU but run multiple streams simultaneously 

* This device has four "big" cores and four "little ones", by limiting the number of threads we ensure that only the big cores are used for the bulk of the compute and avoid the performance hit that typically comes from using big and little cores together. Two threads seem to be the sweet spot, exceptional performance and you can now run multiple streams simultaneously. Four threads only gain 1-2 FPS and is not worth it considering the cost is the opportunity for the device to do more parallel work. 

* Used async functions for reading video and pre-processing, inferencing, post processing and drawing the detection boxes on the frame, the performance uplift is small for one video stream, but at around 10+ FPS for running multiple streams at once. 

Future plans include adding some sort of unique tracking, border crossing, etc., and generating JSON payloads for transmission by protocols like MQTT. I.e., add the features that would needed for an edge deployment. 

![Rockchip RK3588 YOLOv8 NPU](../images/rk3588.gif)
* Trying to do screen capture on top of the two streams grinds things to a halt, so recording with my phone was the best option at the moment. 
* Adding the latency to draw boxes + render the frame costs about 2-3 FPS, so the real on screen FPS is closer to 27-28.
* Both videos are 640 x 360

### Performance
Aided by strict limits on threads to ensure we only use the Rockchip3588 big cores for the bulk of the compute, and asynchronous functions to avoid thread locks and bottlenecks, this implementation of YOLOv8 via the RK3588’s NPU reaches about 48-49 FPS just based on inferencing and ~33 FPS when post-processing is factored in. Another benefit is that since performance only requires one NPU core, and you’re limiting the CPU threads, you can run at 30+ FPS on two standard definition video streams at the same time.  

In comparison, without the threading and asynchronous optimizations, this ran at barely 20 FPS and was using 95-100% of all eight cores. Also: post processing took 20-22ms and was a drag on inferencing, which was in the 37-40 FPS range with post processing and 48-49 FPS without it. Switching to async functions dropped post processing down to 10ms, inferencing FPS was around 48-4 FPS AKA as fast it was without post processing in the old implementation. Additionally, CPU utilization goes as high as 90-100% for 1-2 cores, with the rest typically under 30%. 

### Setup & Technical Details

I used the desktop version of [Joshua Riek's Ubuntu 22.04 distro](https://github.com/Joshua-Riek/ubuntu-rockchip) for Rockchip 3588 devices the operating system, as it delivers better performance than anything else I've tried. 

To run the models on the NPU, you'll need to run the conversion tool (RKNN Toolkit 2) on an x86 machine to run the conversion tool and then use "RKNN Toolkit Lite" to run the models on your Rockchip 3588 device. Suggested approach:

1) [Clone the RKNN-Toolkit2 repo](https://github.com/airockchip/rknn-toolkit2) on the x86 machine you'll use for converting the models AND on your RK3588 device. Make sure it's the most recent repo as they have another with an identical name that is no longer supported. 
2) Once you've cloned the repo, you'll need to go to this [folder](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/packages) and do two things (note: there is a C++ option too, but I haven't tried it yet)
    * Install the requirements.txt for your version of Python, it may fail if you're missing certain dependencies but the error messages are fairly clear so those issues are easy to fix, just know it may take a few tries due to the build dependencies not being there.
    * Install the Python wheel with their conversion tools 
3) Setup your RK3588 device for running the model:
    * Go to packages in the [rknn-toolkit-lite-2 folder](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages) and install the python wheel for your version of Python.
    * Go the the runtime folder in the [rknpu2 folder](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Linux/librknn_api/aarch64) and copy the .so file via sudo to your /usr/lib folder.
4) Once the above is done, just go to the examples folder under rknn-toolkit2 and then try out a few of the conversion examples, you can then do the same on your RK3588 device. "Most" of the examples should run just fine. You can also try the examples on the [RKNN model Zoo](https://github.com/airockchip/rknn_model_zoo)
5) Process wise, the basic steps are: convert the model to ONNX and then use Rockchip's tools to convert the model to RKNN format. 
6) Two other links to keep in mind:
    *[RKNN model Zoo](https://github.com/airockchip/rknn_model_zoo) has models you can use for conversion, but outside of the Rockchip modified ones for YOLO (the aforementioned things moved to CPU)  you can download just them from their original source.
    * You can potentially use [Rockchip's fork](https://github.com/airockchip/ultralytics_yolov8) of Ultralytics YOLOv8 repo to convert your YOLO models to ONNX, but your mileage may vary, e.g., code samples or parameters that don’t work, models that fail to produce inferences or the boxes are nearly off the screen, etc.


### Thoughts on RKNN Tools and Drivers: 

#### Get the negatives out of the way first..

* **Elephant in the room:** closed source .so file coupled with open-source tools that are often buggy and are not updated very often. Plus it can be hard to convert your own trained models for YOLOv5 and v8. Not exactly ideal. I am not an open-source zealot per se, but when you are talking about a device aimed at that intersection between people trying to learn, seasoned engineers prototyping things, homelabbers and experienced makers, open source seems the way to go IMO, not to mention ease of use is critical.

* The tools are "okay" once you work out a few bugs, fix a few items, identify issues where a script is presented as for testing on x86 but is really for running on your RK3588 device, etc., absolutely one of those "hard until it isn't" scenarios. Even with Google appearing to have abandoned Coral, given that the devices are easy to use and readily available, they're a viable alternative for a few years until things in this space improve.

* It's not really a Six TOPS NPU, it's more like [3 x 1 TOPs NPUs IF you're lucky](https://clehaxze.tw/gemlog/2023/07-13-rockchip-npus-and-deploying-scikit-learn-models-on-them.gmi)  

* It's difficult to get the three NPU cores to work together. 

#### For the device overall, there is a plenty of good... 

* Good (and possibly great with C++ and some tweaking) performance, especially when you consider how how small and *"low powered"* this device is. 

* An [open-source driver has been created](https://www.hackster.io/news/tomeu-vizoso-s-open-source-npu-driver-project-does-away-with-the-rockchip-rk3588-s-binary-blob-0153cf723d44) that should be incorporated into Mesa 3d soon, once that happens, we will be able to use TensorFlow Lite delegates to push models to the NPU. I.e., if I were to build an ML project on this device for home or work, my *strong preference* would be to use the open-source driver. Once I finish the item below, I plan to tackle using the open source driver next.

* Yes, the models typically only use one NPU core, but that also means that for certain models you can run three of them in parallel provided the post process steps are light. I'm working on another example that runs a model on each core, meaning: you could monitor and run ML on three different video streams with one of these devices. 

Overall, I think the RKNN tools are best used for models that a) can run entirely on the NPU b) don't require any sort of modification before conversion to RKNN format. 

Finally, support independent open source devs and maintainers however you can, as they're building the tools that will enable Rockchip 3588 devices to reach their full potential.