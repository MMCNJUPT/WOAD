# WOAD: A Wearable Obstacle Avoidance Device for Visually Impaired Individuals with Cross-Modal Learning

This repository contains the source codes and data for our paper reviewed in Nature Communications:

## 1. Data collection and pre-processing

You can access all the data and source codes from [Google Drive Link](https://drive.google.com/file/d/1wmSLaWwfrAhYCl45vA9duIuWpEkHk4q8/view?usp=sharing)

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

The expected output is the depth/ir/rgb data in `bin/x64/testxxxxxxxxxx`.
You can access the demo data from [Google Drive Link](https://drive.google.com/file/d/16nxM4GkWZGpKtzeJkYbadz4rSrCrG172/view?usp=sharing)

The expected run time of the demo will take no longer than 2 minutes on a "normal" desktop computer with the cs30 device.


## 2. RL for adaptive video compression 

You can access all the data and source codes from [Google Drive Link](https://drive.google.com/file/d/100_sZQyzOolvUEu6isxiwMHsYrxk5sC5/view?usp=sharing)

### Environment

* Python 3.8  
* Torch 1.13.0  
* D3rlpy 1.1.1

1. Run `pip install -r requirements.txt` to install all dependencies required in your machine.

2. Import PyTorch with the correct CUDA version.

The installation time will take no longer than 20 minutes on a "normal" desktop computer with good Internet conditions.

### Data availability

All the necessary data for the demo is included in the `./RL_data folder`.

Download the data form [Google Drive Link](https://drive.google.com/file/d/100_sZQyzOolvUEu6isxiwMHsYrxk5sC5/view?usp=sharing) and put it into the `./RL_data folder`.

### Demo
* Run `python train_RL.py` to get policy model without cerebellar reward.
* Run `python creat_MDPDataset_cerebellar_reward.py` to get dataset with cerebellar reward.
* Run `python train_cerebellar_RL.py` to get policy model with cerebellar reward.
* Run `python save_DSAC_onnx_model.py` to save poliicy model.

In the end, you will obtain `policy_cerebellar.onnx`, which will be deployed on the WOAD as C++ code for pre-processing.

The expected run time of the demo will take no longer than 10 minutes on a "normal" desktop computer.


## 3. FPGA deployment 

You can access all the data and source codes from [Google Drive Link](https://drive.google.com/file/d/1PExD1QZmMm3K0I-1pPuamR4yuenDzLP_/view?usp=sharing)

### Environment

* Vscode 1.91.1
* Vscode extension Digital-IDE 0.2.6
* Vivado 2018.3

The installation time will take no longer than 30 minutes on a "normal" desktop computer with good Internet conditions.

### File Structure

```
📦user
 ┣ 📂data
 ┃ ┣ 📂in --------------- Model parameter files
 ┃ ┃ ┣ 📂coe
 ┃ ┃ ┣ 📂fp16
 ┃ ┃ ┗ 📂fp32
 ┃ ┗ 📜map.xdc ---------- Constraints file
 ┣ 📂ip ----------------- Xilinx IP files
 ┃ ┣ 📂clk_25m 
 ┣ 📂sim ---------------- Simulation files
 ┃ ┣ 📂math
 ┃ ┣ 📂RL
 ┃ ┃ ┣ 📂out
 ┃ ┣ 📂VIP
 ┃ ┃ ┣ 📂data
 ┃ ┗ 📜testbench.v
 ┗ 📂src ---------------- Source files
 ┃ ┣ 📂driver
 ┃ ┃ ┗ 📂uart
 ┃ ┣ 📂process
 ┃ ┃ ┗ 📂RL
 ┃ ┃ ┃ ┣ 📂AXI_Lite
 ┃ ┃ ┃ ┣ 📂ROM
 ┃ ┣ 📂system
 ┃ ┣ 📂utils 
 ┃ ┃ ┣ 📂math
 ┃ ┃ ┃ ┣ 📂comb
 ┃ ┃ ┃ ┣ 📂timing
 ┃ ┗ 📜RL_top.v --------- top design
```
### Demo

[Plugin Tutorial](https://sterben.nitcloud.cn/)
```
// property.json
{
	"TOOL_CHAIN": "xilinx",
	"PRJ_NAME": {
		"FPGA": "Tactical-helmet"
	},
	"SOC_MODE": {
		"soc": "none"
	},
	"enableShowlog": false,
	"Device": "xc7z010clg400-1"
}
```
* `launch` ----- to start the whole project
* `build` ------ to build the whole project and finally output the bit stream file
* `program` ---- download the bitstream file to the FPGA/zynq board

4. Cross-Modal obstacle detection : [google drive](https://drive.google.com/file/d/1rUKuZdITwKC5Puv39rheigj6lne3HswW/view?usp=sharing)

5. Smartphone deployment : [google drive](https://drive.google.com/file/d/1Kava0aKGvZWK7KlZPpPcejlNSpcZgbpT/view?usp=sharing)
