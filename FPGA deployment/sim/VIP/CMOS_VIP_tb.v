`timescale 1ns/1ns
module CMOS_VIP_TB;

// `include "../../src/config.v"
`define sim_fun
`ifdef sim_fun
    parameter IMG_HDISP = 1280;	//1280*720
    parameter IMG_VDISP = 720;
    parameter IMG_DATAW = 16;
`else
    parameter IMG_HDISP = 1280;	//60*20
    parameter IMG_VDISP = 720;
    parameter IMG_DATAW = 16;   
`endif

//------------------------------------------
//Generate 24MHz driver clock
reg	clk;
localparam PERIOD2 = 41;		//24MHz
initial begin
    clk = 0;
    forever
        #(PERIOD2/2)
            clk = ~clk;
end

//------------------------------------------
//Generate global reset
reg	rst_n;
task task_reset;
    begin
        rst_n = 0;
        repeat(2) @(negedge clk);
        rst_n = 1;
    end
endtask
wire	clk_cmos = clk;		//24MHz
wire	sys_rst_n = rst_n;

//-----------------------------------------
//CMOS Camera interface and data output simulation
wire			cmos_pclk;				//24MHz CMOS Pixel clock input
wire			cmos_vsync;				//L: vaild, H: invalid
wire			cmos_href;				//H: vaild, L: invalid
wire	[7:0]	cmos_data;				//8 bits cmos data input
CMOS_OUT_tb	#(
    .CMOS_VSYNC_VALID	(1'b1),     //VSYNC = 1
    .IMG_HDISP			(IMG_HDISP),	//(10'd640),	//640*480
    .IMG_VDISP			(IMG_VDISP))	//(10'd480)
u_CMOS_OUT_tb (
    //global reset
    .rst_n				(sys_rst_n),

    //CMOS Camera interface and data output simulation
    .cmos_xclk			(clk_cmos),			//24MHz cmos clock
    .cmos_pclk			(cmos_pclk),		//24MHz when rgb output
    .cmos_vsync			(cmos_vsync),		//L: vaild, H: invalid
    .cmos_href			(cmos_href),		//H: vaild, L: invalid
    .cmos_data			(cmos_data)			//8 bits cmos data input
);

//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
//cmos video image capture
wire			cmos_init_done = 1'b1;	///cmos camera init done
wire			cmos_frame_vsync;	//cmos frame data vsync valid signal
wire			cmos_frame_href;	//cmos frame data href vaild  signal
wire	[15:0]	cmos_frame_data;	//cmos frame data output: {cmos_data[7:0]<<8, cmos_data[7:0]}
wire			cmos_frame_clken;	//cmos frame data output/capture enable clock
wire	[7:0]	cmos_fps_rate;		//cmos image output rate
capturer_rgb565	#(
    .CMOS_FRAME_WAITCNT		(4'd0))				//Wait n fps for steady(OmniVision need 10 Frame)
u_Capture_RGB565 (
    //global clock
    .sys_rstn					(sys_rst_n & cmos_init_done),	//global reset

    //CMOS Sensor Interface
    .cmos_pclk				(cmos_pclk),  		//24MHz CMOS Pixel clock input
    .cmos_data				(cmos_data),		//8 bits cmos data input
    .cmos_vsync				(cmos_vsync),		//L: vaild, H: invalid
    .cmos_href				(cmos_href),		//H: vaild, L: invalid

    //CMOS SYNC Data output
    .cmos_frame_vsync		(cmos_frame_vsync),	//cmos frame data vsync valid signal
    .cmos_frame_href		(cmos_frame_href),	//cmos frame data href vaild  signal
    .cmos_frame_data		(cmos_frame_data),	//cmos frame RGB output: {{R[4:0],G[5:3]}, {G2:0}, B[4:0]}
    .cmos_frame_clken 		(cmos_frame_clken),	//cmos frame data output/capture enable clock

    //user interface
    .cmos_fps_rate			(cmos_fps_rate)		//cmos image output rate
);

Bilinear #(
    .IMG_HDISP_I    ( IMG_HDISP     ),
    .IMG_VDISP_I    ( IMG_VDISP     ),
    .IMG_DATAW 		( IMG_DATAW     ))
u_Bilinear(
    //ports
    .pix_clk            	( cmos_pclk         ),
    .sys_rstn          		( sys_rst_n         ),
    .per_frame_clken        ( cmos_frame_clken  ),
    .per_frame_vsyn 		( cmos_frame_vsync  ),
    .per_frame_href 		( cmos_frame_href   ),
    .per_frame_data 		( cmos_frame_data   )
);

//---------------------------------------------
//testbench of the RTL
task task_sysinit;
    begin
    end
endtask

//----------------------------------------------
initial begin
    task_sysinit;
    task_reset;
end

endmodule

