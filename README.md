# Blur Image Sharpness Assessment

Tensorflow implementation for scoring sharpness of a blur image.

Network structure for transformation network looks:
![network](/assets/network.png)

This repository contains train and test codes for reproduce.
Pretrained network model and dataset will be distributed soon.

--------------------------

## Prerequisites
- tensorflow r1.0 or higher version
- numpy 1.11.0

## Getting Started
### Installation
- Install tensorflow from https://github.com/tensorflow/tensorflow
- Clone this repo:
```bash
git clone https://github.com/xcodebugger/blur_sharpness_assessment.git
cd blur_sharpness_assessment
```

## Training and Test Details
- you need to specify directories for dataset, checkpoint and sample in main.py
- To train a model,  
```bash
python main.py --is_train=True
```
- To test the model,
```bash
python main.py --is_train=False
```
