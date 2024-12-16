# WOAD: A Wearable Obstacle Avoidance Device for Visually Impaired Individuals with Cross-Modal Learning

This repository contains the source codes and data for our paper reviewed in Nature Communications:

For all version details and readme files, please refer to each subfolder.

## 1. Data collection and pre-processing

You can access all the data and source codes from [Google Drive Link](https://drive.google.com/file/d/1wmSLaWwfrAhYCl45vA9duIuWpEkHk4q8/view?usp=sharing)

### Installation
* Windows 11
* gcc 9.4.0
* g++ 9.4.0
* opencv_imgproc440.dll
* SonixCamera.dll
* csreconstruction.dll
* libsynexens3.dll

### Device and Driver
* yunshi RGB-TOF multi-modal sensor CS30 [Tao Bao]([https://drive.google.com/file/d/1wmSLaWwfrAhYCl45vA9duIuWpEkHk4q8/view?usp=sharing](https://item.taobao.com/item.htm?app=weixin&bc_fl_src=share-1041093625490825-2-1&bxsign=scdSBct7CquxvdkHc-S4E8wsiVmuhNnGfVVTh85dPNqTWNuEm2GQXuyIiWm6OeJIZhZQuitZ1GizsHz8k4jBWAEDjeX0n--uILfMn887oqK6haUIPwHAFMrkLiir9kBbMxN&cpp=1&id=667364977514&share_crt_v=1&shareurl=true&short_name=h.TfIBcJHKpILSuyL&sp_tk=Qlo0QTNDY2hKMjQ%3D&spm=a2159r.13376460.0.0&tbSocialPopKey=shareItem&tk=BZ4A3CchJ24&un=de399a3a61f9a38803c6109e527695c7&un_site=0&ut_sk=1.Z1QdewEywXgDAB72G3WraOnc_21646297_1734358135484.TaoPassword-WeiXin.1&wxsign=tbwNg4gAxNttXotScVhB7Pjfc_y-d_L6IWOGGIFFRO_vm_ChpbybceveHYKZd-B-VF9os8gs2QtaHmLMsNRris3Y0IUdXhGskAh-l_3cb3vWoJMTVZlwjGp1x54gl-wEDei))
* cs30-driver v.11

### Demo
* Run the `save_depth_ir_rgb.exe` in `bin/x64`.

The expected output is the depth/ir/rgb data in `bin/x64/xxxxxxxxx`.
You can access the demo data from [Google Drive Link](https://drive.google.com/file/d/1wmSLaWwfrAhYCl45vA9duIuWpEkHk4q8/view?usp=sharing)

The expected run time of the demo will take no longer than 2 minutes on a "normal" desktop computer with the cs30 device.


## 2. RL for adaptive video compression : [google drive](https://drive.google.com/file/d/1qkGsaKZv7PPvwZIAGaCzVvOe59MqIPkk/view?usp=sharing)

3. FPGA deployment : [google drive](https://drive.google.com/file/d/1PExD1QZmMm3K0I-1pPuamR4yuenDzLP_/view?usp=sharing)

4. Cross-Modal obstacle detection : [google drive](https://drive.google.com/file/d/1rUKuZdITwKC5Puv39rheigj6lne3HswW/view?usp=sharing)

5. Smartphone deployment : [google drive](https://drive.google.com/file/d/1Kava0aKGvZWK7KlZPpPcejlNSpcZgbpT/view?usp=sharing)
