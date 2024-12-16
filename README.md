# WOAD: A Wearable Obstacle Avoidance Device for Visually Impaired Individuals with Cross-Modal Learning

This repository contains the source codes and data for our paper reviewed in Nature Communications:

For all version details and readme files, please refer to each subfolder.

## 1. Data collection and pre-processing

You can access all the data and source codes from [Google Drive Link](https://drive.google.com/file/d/1wmSLaWwfrAhYCl45vA9duIuWpEkHk4q8/view?usp=sharing)

### Installation
Environment:
* Windows 11
* Gcc 9.4.0
* G++ 9.4.0
* Opencv_imgproc440.dll
* SonixCamera.dll
* Csreconstruction.dll
* Libsynexens3.dll

### Device and Driver
* Synexens RGB-TOF multi-modal sensor [CS30](https://support.tofsensors.com/product/CS30.html)
* cs30-driver v.11

### Demo
* Run the `save_depth_ir_rgb.exe` in `bin/x64`.

The expected output is the depth/ir/rgb data in `bin/x64/xxxxxxxxx`.
You can access the demo data from [Google Drive Link](https://drive.google.com/file/d/1wmSLaWwfrAhYCl45vA9duIuWpEkHk4q8/view?usp=sharing)

The expected run time of the demo will take no longer than 2 minutes on a "normal" desktop computer with the cs30 device.


## 2. RL for adaptive video compression 

You can access all the data and source codes from [Google Drive Link](https://drive.google.com/file/d/1qkGsaKZv7PPvwZIAGaCzVvOe59MqIPkk/view?usp=sharing)

### Installation

* python 3.8  
* torch 1.13.0  
* d3rlpy 1.1.1  

Data availability

## 3. FPGA deployment : [google drive](https://drive.google.com/file/d/1PExD1QZmMm3K0I-1pPuamR4yuenDzLP_/view?usp=sharing)

4. Cross-Modal obstacle detection : [google drive](https://drive.google.com/file/d/1rUKuZdITwKC5Puv39rheigj6lne3HswW/view?usp=sharing)

5. Smartphone deployment : [google drive](https://drive.google.com/file/d/1Kava0aKGvZWK7KlZPpPcejlNSpcZgbpT/view?usp=sharing)
