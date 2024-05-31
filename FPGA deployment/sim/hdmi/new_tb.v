module new_tb();

parameter MAIN_FRE   = 50; //unit MHz
reg                   sys_clk = 0;
reg                   sys_rst = 0;

always begin
    #(500/MAIN_FRE) sys_clk = ~sys_clk;
end

always begin
    #50 sys_rst = 1;
end

//Instance 
wire [2:0]	TMDS_DATA_P;
wire [2:0]	TMDS_DATA_N;
wire 	TMDS_CLK_P;
wire 	TMDS_CLK_N;
wire 	TMDS_OUT_EN;

hdmi_top #(
	.VIDEO_RATE 		( 25 		))
u_hdmi_top(
	//ports
	.clk_50m     		( sys_clk     		),
	.rst         		( sys_rst         	),
	.TMDS_DATA_P 		( TMDS_DATA_P 		),
	.TMDS_DATA_N 		( TMDS_DATA_N 		),
	.TMDS_CLK_P  		( TMDS_CLK_P  		),
	.TMDS_CLK_N  		( TMDS_CLK_N  		),
	.TMDS_OUT_EN 		( TMDS_OUT_EN 		)
);

endmodule  //TOP
