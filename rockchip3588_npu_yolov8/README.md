## YOLOv8 Object Detection on the NPU of a Rockchip 3588 System on Chip (SOC)

Relatively simple starter implementation/example of using the NPU of a Rockchip 3588 to accelerate YOLOv8. Plan is to optimize the code a bit more for speed and then add some tracking and counting features, given how you have to implement YOLOv8 via Rockchip's RKNN format the built in tracking and counting features of YOLO aren't available so I'll have to either *"build a bridge"* to said features, rebuild them myself or some combination of both. 


### Performance
* If you just look at inferencing, it's about 35-40 FPS for a 640 x 360 image, however, to run YOLO models on this NPU requires you to remove some items from the original neural network (around generating predictions, probabilities and the like) and move them to the CPU, which adds another 20ms. Meaning: the real FPS is around 20, goal is to figure out how to slice down a bit. 
* It's worth noting that if you skip the post processing steps the FPS jumps to close to 50, so reducing the amount compute needed for post processing will speed this up tremendously. 


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
    *[RKNN model Zoo](https://github.com/airockchip/rknn_model_zoo) has models you can use for conversion, but outside of the Rockchip modified ones for YOLO (the aforementioned things moved to CPU)  you can download just them from their original source.
    * You can potentially use [Rockchip's fork](https://github.com/airockchip/ultralytics_yolov8) of Ultralytics YOLOv8 repo to convert your YOLO models to ONNX, but your mileage may vary, e.g., code samples or parameters that don’t work, models that fail to produce inferences or the boxes are nearly off the screen, etc. 


### Thoughts on RKNN Tools and Drivers: 

#### Get the negatives out of the way first..

* **Elephant in the room:** closed source .so file coupled with open-source tools that are often buggy and are not updated very often. Plus it can be hard to convert your own trained models for YOLOv5 and v8. Not exactly ideal. I am not an open-source zealot per se, but when you are talking about a device aimed at that intersection between people trying to learn, seasoned engineers prototyping things, homelabbers and experienced makers, open source seems the way to go IMO.

* The tools are "okay" once you work out a few bugs, fix a few items, identify issues where a script is presented as for testing on x86 but is really for running on your RK3588 device, etc., absolutely one of those "hard until it isn't" scenarios. So,.. you know, choose wisely, especially considering that Google Coral devices make it easy for you to convert and deploy TensorFlow Lite models with relatively little hassle.

* It's not really a Six TOPS NPU, it's more like [3 x 1 TOPs NPUs IF you're lucky](https://clehaxze.tw/gemlog/2023/07-13-rockchip-npus-and-deploying-scikit-learn-models-on-them.gmi)  

* It's difficult to get the three NPU cores to work together. 

#### For the device overall, there is a plenty of good... 

* Good (and possibly great with C++ and some tweaking) performance given how small and low powered this device is. 

* An [open-source driver has been created](https://www.hackster.io/news/tomeu-vizoso-s-open-source-npu-driver-project-does-away-with-the-rockchip-rk3588-s-binary-blob-0153cf723d44) that should be incorporated into Mesa 3d soon, once that happens, we will be able to use TensorFlow Lite delegates to push models to the NPU. I.e., if I were to build an ML project on this device for home or work, I would use the open-source driver. Onec I finish the item below, I plan to tackle using the open source driver next.

* Yes, the models typically only use one NPU core, but that also means that for certain models you can run three of them in parallel provided the post process steps are light. I'm working on another example that runs a model on each core, meaning: you could monitor and run ML on three different video streams with one of these devices. 

Overall, I think the RKNN tools are best used for models that a) can run entirely on the NPU b) don't require any sort of modification before conversion to RKNN format. 