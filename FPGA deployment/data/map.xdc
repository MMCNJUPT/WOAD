# system interface
## 7020
# set_property -dict {PACKAGE_PIN E17 IOSTANDARD LVCMOS33} [get_ports sys_rstn]
# set_property -dict {PACKAGE_PIN K17 IOSTANDARD LVCMOS33} [get_ports sys_clk]

## 7100
# set_property -dict {PACKAGE_PIN AF18 IOSTANDARD LVCMOS33} [get_ports sys_rstn]
# set_property -dict {PACKAGE_PIN F9 IOSTANDARD DIFF_SSTL15} [get_ports sys_clk_p]
# create_clock -period 5.000 -name sys_clk_p -waveform {0.000 2.500} [get_ports sys_clk_p]

# LED0~1 interface (7100 @ ailinx)
# set_property -dict {PACKAGE_PIN AJ16 IOSTANDARD LVCMOS33} [get_ports i2c_config_done]
# set_property -dict {PACKAGE_PIN AK16 IOSTANDARD LVCMOS33} [get_ports {led[1]}]
# set_property -dict {PACKAGE_PIN AE16 IOSTANDARD LVCMOS33} [get_ports {led[2]}]
# set_property -dict {PACKAGE_PIN AE15 IOSTANDARD LVCMOS33} [get_ports {led[3]}]

# CMOS interface (OV5640 @ 7100-ailinx)
# set_property -dict {PACKAGE_PIN AK13 IOSTANDARD LVCMOS33 PULLUP true} [get_ports {cmos_sdat}]
# set_property -dict {PACKAGE_PIN AF15 IOSTANDARD LVCMOS33 PULLUP true} [get_ports {cmos_sclk}]

# set_property -dict {PACKAGE_PIN AB17 IOSTANDARD LVCMOS33} [get_ports {cmos_rstn}]
# set_property -dict {PACKAGE_PIN AB14 IOSTANDARD LVCMOS33} [get_ports {cmos_data[7]}]
# set_property -dict {PACKAGE_PIN AB15 IOSTANDARD LVCMOS33} [get_ports {cmos_data[6]}]
# set_property -dict {PACKAGE_PIN AC16 IOSTANDARD LVCMOS33} [get_ports {cmos_data[5]}]
# set_property -dict {PACKAGE_PIN AD15 IOSTANDARD LVCMOS33} [get_ports {cmos_data[4]}]
# set_property -dict {PACKAGE_PIN AG15 IOSTANDARD LVCMOS33} [get_ports {cmos_data[3]}]
# set_property -dict {PACKAGE_PIN AF14 IOSTANDARD LVCMOS33} [get_ports {cmos_data[2]}]
# set_property -dict {PACKAGE_PIN AG14 IOSTANDARD LVCMOS33} [get_ports {cmos_data[1]}]
# set_property -dict {PACKAGE_PIN AA14 IOSTANDARD LVCMOS33} [get_ports {cmos_data[0]}]
# set_property -dict {PACKAGE_PIN AC17 IOSTANDARD LVCMOS33} [get_ports {cmos_data[1]}]
# set_property -dict {PACKAGE_PIN AD16 IOSTANDARD LVCMOS33} [get_ports {cmos_data[0]}]

# set_property -dict {PACKAGE_PIN AA1 IOSTANDARD LVCMOS33} [get_ports {cmos_pclk}]
# set_property -dict {PACKAGE_PIN AJ1 IOSTANDARD LVCMOS33} [get_ports {cmos_href}]
# set_property -dict {PACKAGE_PIN AK1 IOSTANDARD LVCMOS33} [get_ports {cmos_vsync}]

# HDMI interface (7020 @ MicroPhase)
# set_property IOSTANDARD LVCMOS33 [get_ports HDMI_OUT_EN]
# set_property PACKAGE_PIN F17 [get_ports HDMI_OUT_EN]

# set_property PACKAGE_PIN H16 [get_ports TMDS_CLK_P]
# set_property PACKAGE_PIN D19 [get_ports {TMDS_DATA_P[0]}]
# set_property PACKAGE_PIN C20 [get_ports {TMDS_DATA_P[1]}]
# set_property PACKAGE_PIN B19 [get_ports {TMDS_DATA_P[2]}]

## 7010 RL
set_property -dict {PACKAGE_PIN T10 IOSTANDARD LVCMOS33} [get_ports uart_tx]

set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]

## HDMI interface (7010 @ ALINX)
#set_property -dict {PACKAGE_PIN Y18 IOSTANDARD LVCMOS33} [get_ports clk_50m]
#set_property -dict {PACKAGE_PIN M13 IOSTANDARD LVCMOS33} [get_ports rst]

#set_property IOSTANDARD LVCMOS33 [get_ports TMDS_OUT_EN]
#set_property PACKAGE_PIN M6 [get_ports TMDS_OUT_EN]

#set_property PACKAGE_PIN E1 [get_ports TMDS_CLK_P]
#set_property PACKAGE_PIN G1 [get_ports {TMDS_DATA_P[0]}]
#set_property PACKAGE_PIN H2 [get_ports {TMDS_DATA_P[1]}]
#set_property PACKAGE_PIN K1 [get_ports {TMDS_DATA_P[2]}]

#set_property IOSTANDARD TMDS_33 [get_ports {TMDS_DATA_P[2]}]
#set_property IOSTANDARD TMDS_33 [get_ports {TMDS_DATA_P[1]}]
#set_property IOSTANDARD TMDS_33 [get_ports {TMDS_DATA_P[0]}]
#set_property IOSTANDARD TMDS_33 [get_ports TMDS_CLK_P]

#set_property PACKAGE_PIN P17 [get_ports clk_test]
#set_property IOSTANDARD LVCMOS33 [get_ports clk_test]





set_property PACKAGE_PIN G11 [get_ports out]
set_property PACKAGE_PIN G15 [get_ports ovalid]
set_property PACKAGE_PIN H14 [get_ports sys_rst]
set_property PACKAGE_PIN J16 [get_ports uart_rx]
set_property PACKAGE_PIN C11 [get_ports sys_clk]
set_property LOC RAMB36_X1Y17 [get_cells u_RL/u_GEMMR2/u_gemm2B/out_reg_20]
set_property LOC RAMB36_X1Y18 [get_cells u_RL/u_GEMMR2/u_gemm2B/out_reg_18]
set_property LOC RAMB36_X1Y19 [get_cells u_RL/u_GEMMR2/u_gemm2B/out_reg_26]
set_property LOC RAMB36_X2Y16 [get_cells u_RL/u_GEMMR2/u_gemm2B/out_reg_7]
set_property LOC RAMB36_X2Y17 [get_cells u_RL/u_GEMMR2/u_gemm2B/out_reg_4]
set_property LOC RAMB36_X2Y18 [get_cells u_RL/u_GEMMR2/u_gemm2B/out_reg_3]
set_property LOC RAMB36_X2Y19 [get_cells u_RL/u_GEMMR2/u_gemm2B/out_reg_1]
set_property BEL DSP48E1 [get_cells {u_RL/u_GEMMR2/u_vector/Mult_array[1].u_mult/fraction0}]
set_property BEL DSP48E1 [get_cells {u_RL/u_GEMMR2/u_vector/Mult_array[2].u_mult/fraction0}]

create_debug_core u_ila_0 ila
set_property ALL_PROBE_SAME_MU true [get_debug_cores u_ila_0]
set_property ALL_PROBE_SAME_MU_CNT 1 [get_debug_cores u_ila_0]
set_property C_ADV_TRIGGER false [get_debug_cores u_ila_0]
set_property C_DATA_DEPTH 2048 [get_debug_cores u_ila_0]
set_property C_EN_STRG_QUAL false [get_debug_cores u_ila_0]
set_property C_INPUT_PIPE_STAGES 0 [get_debug_cores u_ila_0]
set_property C_TRIGIN_EN false [get_debug_cores u_ila_0]
set_property C_TRIGOUT_EN false [get_debug_cores u_ila_0]
set_property port_width 1 [get_debug_ports u_ila_0/clk]
connect_debug_port u_ila_0/clk [get_nets [list sys_clk_IBUF_BUFG]]
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe0]
set_property port_width 16 [get_debug_ports u_ila_0/probe0]
connect_debug_port u_ila_0/probe0 [get_nets [list {u_RL/others_a[0]} {u_RL/others_a[1]} {u_RL/others_a[2]} {u_RL/others_a[3]} {u_RL/others_a[4]} {u_RL/others_a[5]} {u_RL/others_a[6]} {u_RL/others_a[7]} {u_RL/others_a[8]} {u_RL/others_a[9]} {u_RL/others_a[10]} {u_RL/others_a[11]} {u_RL/others_a[12]} {u_RL/others_a[13]} {u_RL/others_a[14]} {u_RL/others_a[15]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe1]
set_property port_width 16 [get_debug_ports u_ila_0/probe1]
connect_debug_port u_ila_0/probe1 [get_nets [list {u_RL/others_b[0]} {u_RL/others_b[1]} {u_RL/others_b[2]} {u_RL/others_b[3]} {u_RL/others_b[4]} {u_RL/others_b[5]} {u_RL/others_b[6]} {u_RL/others_b[7]} {u_RL/others_b[8]} {u_RL/others_b[9]} {u_RL/others_b[10]} {u_RL/others_b[11]} {u_RL/others_b[12]} {u_RL/others_b[13]} {u_RL/others_b[14]} {u_RL/others_b[15]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe2]
set_property port_width 32 [get_debug_ports u_ila_0/probe2]
connect_debug_port u_ila_0/probe2 [get_nets [list {data_out_0[0]} {data_out_0[1]} {data_out_0[2]} {data_out_0[3]} {data_out_0[4]} {data_out_0[5]} {data_out_0[6]} {data_out_0[7]} {data_out_0[8]} {data_out_0[9]} {data_out_0[10]} {data_out_0[11]} {data_out_0[12]} {data_out_0[13]} {data_out_0[14]} {data_out_0[15]} {data_out_0[16]} {data_out_0[17]} {data_out_0[18]} {data_out_0[19]} {data_out_0[20]} {data_out_0[21]} {data_out_0[22]} {data_out_0[23]} {data_out_0[24]} {data_out_0[25]} {data_out_0[26]} {data_out_0[27]} {data_out_0[28]} {data_out_0[29]} {data_out_0[30]} {data_out_0[31]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe3]
set_property port_width 1 [get_debug_ports u_ila_0/probe3]
connect_debug_port u_ila_0/probe3 [get_nets [list uart_done]]
set_property C_CLK_INPUT_FREQ_HZ 300000000 [get_debug_cores dbg_hub]
set_property C_ENABLE_CLK_DIVIDER false [get_debug_cores dbg_hub]
set_property C_USER_SCAN_CHAIN 1 [get_debug_cores dbg_hub]
connect_debug_port dbg_hub/clk [get_nets sys_clk_IBUF_BUFG]
