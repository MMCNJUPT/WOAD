module hdmi_tb();

parameter DATA_WIDTH = 8;
parameter ADDR_WIDTH = 8;
parameter MAIN_FRE   = 100; //unit MHz
reg                   clk_pix = 0;
reg                   clk_pix_x5 = 0;
reg                   sys_rst = 1;
reg [DATA_WIDTH-1:0]  data = 0;
reg [ADDR_WIDTH-1:0]  addr = 0;

always begin
    #(500/50) clk_pix = ~clk_pix;
end

always begin
    #(500/250) clk_pix_x5 = ~clk_pix_x5;
end

always begin
    #50 sys_rst = 0;
end

//Instance 
wire [2:0]	TMDS_DATA_P;
wire [2:0]	TMDS_DATA_N;
wire 	TMDS_CLK_P;
wire 	TMDS_CLK_N;
wire 	HDMI_OUT_EN;

wire Hsync;
wire Vsync;
wire ready;
wire [7:0] Red;
wire [7:0] Green;
wire [7:0] Blue;
vga_gen VGA_TOP_INST(
    .RED(Red),
    .GREEN(Green),
    .BLUE(Blue),
    .HSYNC(Hsync),
    .VSYNC(Vsync),
    .READY(ready),

    .CLK(clk_pix),  // for 1280 x 720
    .RST_N(rstn)
);

HDMI_out u_HDMI_out(
	//ports
	.clk_pixel    		( clk_pix    		),
	.clk_pixel_x5 		( clk_pix_x5 		),
	.rst          		( !ready          	),
	.RED          		( Red          		),
	.GREEN        		( Green        		),
	.BLUE         		( Blue         		),
	.HSYNC        		( Hsync        		),
	.VSYNC        		( Vsync        		),
	.TMDS_DATA_P  		( TMDS_DATA_P  		),
	.TMDS_DATA_N  		( TMDS_DATA_N  		),
	.TMDS_CLK_P   		( TMDS_CLK_P   		),
	.TMDS_CLK_N   		( TMDS_CLK_N   		),
	.TMDS_OUT_EN  		( TMDS_OUT_EN  		)
);

endmodule  //TOP
