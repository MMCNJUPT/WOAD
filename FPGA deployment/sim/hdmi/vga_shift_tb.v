module vga_shift_tb();

parameter MAIN_FRE   = 100; //unit MHz
reg                   sys_clk = 0;
reg                   sys_rst = 1;

always begin
    #(500/MAIN_FRE) sys_clk = ~sys_clk;
end

always begin
    #50 sys_rst = 0;
end

//Instance 
wire 	vpg_de;
wire 	vpg_hs;
wire 	vpg_vs;
wire [7:0]	rgb_r;
wire [7:0]	rgb_g;
wire [7:0]	rgb_b;

vga_shift #(
	.H_TOTAL  		( 2200-1 		),
	.H_SYNC   		( 44-1   		),
	.H_START  		( 190-1  		),
	.H_END    		( 2110-1 		),
	.V_TOTAL  		( 1125-1 		),
	.V_SYNC   		( 5-1    		),
	.V_START  		( 41-1   		),
	.V_END    		( 1121-1 		),
	.SQUARE_X 		( 500    		),
	.SQUARE_Y 		( 500    		),
	.SCREEN_X 		( 1920   		),
	.SCREEN_Y 		( 1080   		))
u_vga_shift(
	//ports
	.rst      		( sys_rst      	),
	.vpg_pclk 		( sys_clk 		),
	.vpg_de   		( vpg_de   		),
	.vpg_hs   		( vpg_hs   		),
	.vpg_vs   		( vpg_vs   		),
	.rgb_r    		( rgb_r    		),
	.rgb_g    		( rgb_g    		),
	.rgb_b    		( rgb_b    		)
);

endmodule  //TOP
