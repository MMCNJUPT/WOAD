module  top (
	//global clock
	input sys_clk_p,
    input sys_clk_n,
	input sys_rstn,

	//cmos interface
	output			  cmos_sclk,		//cmos i2c clock
	inout			  cmos_sdat,		//cmos i2c data
	// input			  cmos_pclk,		//cmos pxiel clock
	// output			  cmos_xclk,		//cmos externl clock
	// input			  cmos_vsync,		//cmos vsync
	// input			  cmos_href,		//cmos hsync refrence
	// input	[7:0]	  cmos_data,		//cmos data
	// output		      cmos_reset,		//cmos control 0, reset output
	// output			  cmos_pwdn,		//cmos control 1, pwdn output
	// output   		  cmos_frex,		//cmos control 2, frex output
	// input			  cmos_strobe,		//cmos control 3, strobe
  
	//HDMI interface
    // output  [2:0]     TMDS_DATA_P,
    // output  [2:0]     TMDS_DATA_N,

    // output            TMDS_CLK_P,
    // output            TMDS_CLK_N,

    // output            HDMI_OUT_EN

    output  i2c_config_done
);

wire sys_clk;
IBUFDS #(
    .DIFF_TERM("FALSE"),       // Differential Termination
    .IBUF_LOW_PWR("TRUE"),     // Low power="TRUE", Highest performance="FALSE" 
    .IOSTANDARD("DEFAULT")     // Specify the input I/O standard
) IBUFDS_inst (
    .O(sys_clk),    // Buffer output
    .I(sys_clk_p),  // Diff_p buffer input (connect directly to top-level port)
    .IB(sys_clk_n)  // Diff_n buffer input (connect directly to top-level port)
);

cmos_config #(
	.CMOS_TYPE  		( "ov5640" 		),
	.DATA_TYPE  		( "rgb565" 		),
	.CLOCK_MAIN 		( 200_000  		))
u_cmos_config(
	//ports
	.sys_clk         		( sys_clk         		),
	.sys_rstn        		( sys_rstn        		),
	.cmos_sclk       		( cmos_sclk       		),
	.cmos_sdat       		( cmos_sdat       		),
	.i2c_config_done 		( i2c_config_done 		)
);

endmodule  //module_name