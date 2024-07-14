# Development Environment

- vscode 1.91.1
- vscode extension Digital-IDE 0.2.6
- vivado 2018.3

# File Structure

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

# Quick Start
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

1. launch ------ to start the whole project
2. build ------- to build the whole project and finally output the bit stream file
3. program ----- download the bitstream file to the FPGA/zynq board