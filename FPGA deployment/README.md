You can access all the data and source codes from [Google Drive Link](https://drive.google.com/file/d/1PExD1QZmMm3K0I-1pPuamR4yuenDzLP_/view?usp=sharing)

### Environment

* Vscode 1.91.1
* Vscode extension Digital-IDE 0.2.6
* Vivado 2018.3

The installation time will take no longer than 30 minutes on a "normal" desktop computer with good Internet conditions.

### Device
* FPGA/zynq [xc7a35tftg256](https://www.amd.com/zh-cn/products/adaptive-socs-and-fpgas/fpga/artix-7.html)

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
