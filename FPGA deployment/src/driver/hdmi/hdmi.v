`timescale 1ns / 1ps

// (c) fpga4fun.com & KNJN LLC 2013
// edited/updated for vivado 2020.1 by Dominic Meads 10/2020

////////////////////////////////////////////////////////////////////////
module HDMI_out(
    input  clk_pixel,     // 25MHz
    input  clk_pixel_x5,  // 250MHz
    input  rst,

    input [7:0] RED,
    input [7:0] GREEN,
    input [7:0] BLUE,
    input       HSYNC,
    input       VSYNC,
    input       video_de,

    // HMDI Pin I/O
    output [2:0] TMDS_DATA_P,
    output [2:0] TMDS_DATA_N,

    output TMDS_CLK_P,
    output TMDS_CLK_N,

    output TMDS_OUT_EN
);
    assign TMDS_OUT_EN = 1;
    
    /******************************** HDMI OUT ********************************/
    // 8b/10b encoding for transmission
    wire reset;

    asyn_rst_syn reset_syn(
    .clk                (clk_pixel),          
    .reset_n            (rst),      

    .syn_reset          (reset)    
    );

    wire [9:0]	r_tmds;
    dvi_encoder encoder_r (
    .clkin      (clk_pixel),
    .rstin	    (reset),
    
    .din		(RED),
    .c0			(1'b0),
    .c1			(1'b0),
    .de			(video_de),
    .dout		(r_tmds)
    ) ;

    wire [9:0]	g_tmds;
    dvi_encoder encoder_g (
    .clkin      (clk_pixel),
    .rstin	    (reset),
    
    .din		(GREEN),
    .c0			(1'b0),
    .c1			(1'b0),
    .de			(video_de),
    .dout		(g_tmds)
    );

    wire [9:0]	b_tmds;
    dvi_encoder encoder_b (
    .clkin      (clk_pixel),
    .rstin	    (reset),
    
    .din        (BLUE),
    .c0			(HSYNC),
    .c1			(VSYNC),
    .de			(video_de),
    .dout		(b_tmds)
    );

    wire [2:0] tmds;
    wire tmds_clock;
    oserializer #(
        .VIDEO_RATE(25)) 
    u_oserializer(
        .clk_pixel(clk_pixel), 
        .clk_pixel_x5(clk_pixel_x5), 
        .reset(reset), 
        .red(r_tmds),
        .green(g_tmds),
        .blue(b_tmds),
        .tmds(tmds), 
        .tmds_clock(tmds_clock)
    );

    OBUFDS #(
        .IOSTANDARD("TMDS_33"), // Specify the output I/O standard
        .SLEW("SLOW"))          // Specify the output slew rate
    OBUFDS_red (
        .O(TMDS_DATA_P[2]),     // Diff_p output (connect directly to top-level port)
        .OB(TMDS_DATA_N[2]),    // Diff_n output (connect directly to top-level port)
        .I(tmds[2])   // Buffer input
    );

    OBUFDS #(
        .IOSTANDARD("TMDS_33"), // Specify the output I/O standard
        .SLEW("SLOW"))          // Specify the output slew rate
    OBUFDS_green (
        .O(TMDS_DATA_P[1]),     // Diff_p output (connect directly to top-level port)
        .OB(TMDS_DATA_N[1]),    // Diff_n output (connect directly to top-level port)
        .I(tmds[1]) // Buffer input
    );

    OBUFDS #(
        .IOSTANDARD("TMDS_33"), // Specify the output I/O standard
        .SLEW("SLOW"))          // Specify the output slew rate
    OBUFDS_blue (
        .O(TMDS_DATA_P[0]),     // Diff_p output (connect directly to top-level port)
        .OB(TMDS_DATA_N[0]),    // Diff_n output (connect directly to top-level port)
        .I(tmds[0])  // Buffer input
    );

    OBUFDS #(
        .IOSTANDARD("TMDS_33"), // Specify the output I/O standard
        .SLEW("SLOW"))          // Specify the output slew rate
    OBUFDS_clock (
        .O(TMDS_CLK_P),     // Diff_p output (connect directly to top-level port)
        .OB(TMDS_CLK_N),    // Diff_n output (connect directly to top-level port)
        .I(tmds_clock)          // Buffer input
    );

endmodule  // HDMI_out
