## YOLOv8 Object Detection on the NPU of a Rockchip 3588 System on Chip (SOC)

Relatively simple starter implementation/example of using the NPU of a Rockchip 3588 to accelerate YOLOv8. Plan is to optimize the code a bit more for speed and then add some tracking and counting features, given how you have to implement YOLOv8 via Rockchip's RKNN format the built in tracking and counting features of YOLO aren't available so I'll have to either *"build a bridge"* to said features, rebuild them myself or some combination of both. 


### Performance
* If you just look at inferencing, it's about 35-40 FPS for a 640 x 360 image, however, to run YOLO models on this NPU requires you to remove some items from the original neural network (around generating predictions, probabilities and the like) and move them to the CPU, which adds another 20ms. Meaning: the real FPS is around 20, goal is to figure out how to slice down a bit. 
* It's worth noting that if you skip the post processing steps the FPS jumps to close to 50, just need to figure out a way to reduce the compute requirements of the post processing steps.


### Setup

To run this yourself you need to run the main tool on an x86 machine to run the conversion tool and then run the "Toolkit lite" on your Rockchip 3588 device. Suggested approach:

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
    *[RKNN model Zoo](https://github.com/airockchip/rknn_model_zoo) has models you can use for conversion, but outside of the Rockchip modified ones for YOLO (the aforementioned things moved to CPU)  you can download just from the original source.
    * You can potentially use [Rockchip's fork](https://github.com/airockchip/ultralytics_yolov8) of Ultralytics YOLOv8 repo to convert your YOLO models to ONNX, but your mileage may vary, I've gotten it to work and had it fail too, it's a pain. 


### So, my thoughts:

* The models usually run on only one NPU core, I'm working on another example that runs a model on each core, meaning: you could monitor three cameras on one of these devices. I.e., smaller, simpler models on this NPU are a great option. 
* The tools are "okay" once you work out a few bugs, fix a few items, identify issues where a script is presented as for testing on x86 but is really for running on your RK3588 device, ec., so... you know, choose wisely, especially when Google Coral exists.
* An [open source driver has been created](https://www.hackster.io/news/tomeu-vizoso-s-open-source-npu-driver-project-does-away-with-the-rockchip-rk3588-s-binary-blob-0153cf723d44), which should be incorporated into Mesa 3d soon, once that happens we can use TensorFlow Lite delegates to use the NPU. This was a good learning experience/point of comparison for other options, but for future projects and/or anything work related I plan to use the open source driver.
*  Good (and possibly great with C++ and some tweaking) performance given how small and low powered this device is, but, I'd probably find another option for deploying YOLO at the edge in a work situation. Google's Coral devices are basically sitting right there, you can just convert a TF lite model and be on your way.
* This is probably a better option for scenarios where the RKNN tools can convert a model as is with no tinkering, fewer steps/points of failure.


