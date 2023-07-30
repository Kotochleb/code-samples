# code-samples

This repository contains code samples

### [CUDA_HOG_SVM](./CUDA_HOG_SVM)

Is an implementation of Histogram of Oriented Gradient with Support Vector Machine in CUDA. Created for CUDA programming course at AGH UST.

### [dataset_creator](./dataset_creator)

Is a simple library to generate [COCO](https://cocodataset.org/) format datasets out of known labels within [CARLA](https://carla.org/) simulation engine. Created for computer vision in autonomous vehicles course at AGH UST.

### [dc_motor_lqr](./dc_motor_lqr)

Raspberry pi pico MicroROS firmware with data collection and processing scripts for capstone project at AGH UST. Based on measured data and parameters from datasheet system is accurately identified. Later Kalman Filter, LQR, LQI and LQG controllers are examined on this system.

### [traffic_light_detector](./traffic_light_detector)

Short computer vision code samples created for computer vision in autonomous vehicles course at AGH UST. Meant to be run with [CARLA](https://carla.org/) simulation engine.
- [ClassicDetector.py](./traffic_light_detector/ClassicDetector.py) is a color-space-based approach of traffic light detection. Uses HSV and YCbCr color thresholding and basic geometry properties of expected object to be detected.
- [faster_rcnn.ipynb](./traffic_light_detector/faster_rcnn.ipynb) is transfer learning of Faster RCNN in order to learn it to detect traffic lights. Trained using own dataset created with [dataset_creator](./dataset_creator). Unfortunately performance was poor.
- [svm_train.ipynb](./traffic_light_detector/svm_train.ipynb) creates augumented dataset and trains Support Vector Machine on Histogram of Oriented Gradients.
- [SVMDetector.py](./traffic_light_detector/SVMDetector.py) expands [ClassicDetector.py](./traffic_light_detector/ClassicDetector.py) by adding detection validation with HOG+SVM for better robustness.

[collect.py](./traffic_light_detector/collect.py) and [drive.py](./traffic_light_detector/drive.py) were created for dataset creating for HOG+SVM and for runtime evaluation of algorithms.