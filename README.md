# WOAD: A Wearable Obstacle Avoidance Device for Visually Impaired Individuals with Cross-Modal Learning

This repository contains the source codes and data for our paper reviewed in Nature Communications:

## 1. Data collection and pre-processing

You can access all the data and source codes from [Google Drive Link]()

### Environment
* Windows 11
* Gcc 9.4.0
* G++ 9.4.0
* Opencv_imgproc440.dll
* SonixCamera.dll
* Csreconstruction.dll
* Libsynexens3.dll

The installation time will take no longer than 60 minutes on a "normal" desktop computer with good Internet conditions.

### Device and Driver
* Synexens RGB-TOF multi-modal sensor [CS30](https://support.tofsensors.com/product/CS30.html)
* cs30-driver v.11

### Demo
* Run the `save_depth_ir_rgb.exe` in `bin/x64`.

The expected output is the depth/ir/rgb data in `bin/x64/xxxxxxxxx`.
You can access the demo data from [Google Drive Link]()

The expected run time of the demo will take no longer than 2 minutes on a "normal" desktop computer with the cs30 device.


## 2. RL for adaptive video compression 

You can access all the data and source codes from [Google Drive Link]()

### Installation

* Python 3.8  
* Torch 1.13.0  
* D3rlpy 1.1.1  

The installation time will take no longer than 30 minutes on a "normal" desktop computer with good Internet conditions.

### Data availability

All the necessary data for the demo is included in the `./RL_data folder`. Please obtain all the data from [Google Drive Link]()

### Demo
* Run `python train_RL.py` to get policy model without cerebellar reward.
* Run `python creat_MDPDataset_cerebellar_reward.py` to get dataset with cerebellar reward.
* Run `python train_cerebellar_RL.py` to get policy model with cerebellar reward.
* Run `python save_DSAC_onnx_model.py` to save poliicy model.

In the end, you will obtain `policy_cerebellar.onnx`, which will be deployed on the WOAD as C++ code for pre-processing.

## 3. FPGA deployment 

You can access all the data and source codes from [Google Drive Link](https://drive.google.com/file/d/1PExD1QZmMm3K0I-1pPuamR4yuenDzLP_/view?usp=sharing)

4. Cross-Modal obstacle detection : [google drive](https://drive.google.com/file/d/1rUKuZdITwKC5Puv39rheigj6lne3HswW/view?usp=sharing)

5. Smartphone deployment : [google drive](https://drive.google.com/file/d/1Kava0aKGvZWK7KlZPpPcejlNSpcZgbpT/view?usp=sharing)
