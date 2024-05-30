module hdmi_top #(
    parameter VIDEO_RATE = 25
) (
    input        clk_50m,
    input        rst,

    output [2:0] TMDS_DATA_P,
    output [2:0] TMDS_DATA_N,

    output       TMDS_CLK_P,
    output       TMDS_CLK_N,

    output       TMDS_OUT_EN
);

localparam CNT = 50/VIDEO_RATE - 2;

//wire clk_in_clk_gen;

(* mark_debug = "true" *)wire  clk1x;
(* mark_debug = "true" *)wire  clk5x;

wire locked;

clk_125m instance_name
   (
    // Clock out ports
    .clk_out1(clk1x),     // output clk_out1
    .clk_out2(clk5x),     // output clk_out2
    // Status and control signals
    .reset(~rst), // input reset
    .locked(locked),       // output locked
   // Clock in ports
    .clk_in1(clk_50m));      // input clk_in1

wire  [23:0] video_rgb;
wire 	      vpg_de;
wire 	      vpg_hs;
wire 	      vpg_vs;

wire  [10:0]  pixel_xpos_w;
wire  [10:0]  pixel_ypos_w;
wire  [23:0]  pixel_data_w;

video_driver u_video_driver(
    .pixel_clk      (clk1x),
    .sys_rst_n      (rst),

    .video_hs       (vpg_hs),
    .video_vs       (vpg_vs),
    .video_de       (vpg_de),
    .video_rgb      (video_rgb),

    .pixel_xpos     (pixel_xpos_w),
    .pixel_ypos     (pixel_ypos_w),
    .pixel_data     (pixel_data_w)
    );

video_display  u_video_display(
    .pixel_clk      (clk1x),
    .sys_rst_n      (rst),

    .pixel_xpos     (pixel_xpos_w),
    .pixel_ypos     (pixel_ypos_w),
    .pixel_data     (pixel_data_w)
    );

wire [2:0] tmds_data_p;
wire [2:0] tmds_data_n;
wire       tmds_clk_p;
wire       tmds_clk_n;
wire       tmds_out_en;

assign TMDS_DATA_P = tmds_data_p;
assign TMDS_DATA_N = tmds_data_n;
assign TMDS_CLK_P  = tmds_clk_p;
assign TMDS_CLK_N  = tmds_clk_n;
assign TMDS_OUT_EN = tmds_out_en;

HDMI_out u_HDMI_out(
    .clk_pixel          (clk1x),     // 25MHz
    .clk_pixel_x5       (clk5x),  // 250MHz
    .rst                (rst && locked),

    .RED                (video_rgb[23:16]),
    .GREEN              (video_rgb[15:8]),
    .BLUE               (video_rgb[7:0]),
    .HSYNC              (vpg_hs),
    .VSYNC              (vpg_vs),
    .video_de           (vpg_de),
    // HMDI Pin I/O
    .TMDS_DATA_P        (tmds_data_p),
    .TMDS_DATA_N        (tmds_data_n),

    .TMDS_CLK_P         (tmds_clk_p),
    .TMDS_CLK_N         (tmds_clk_n),

    .TMDS_OUT_EN        (tmds_out_en)
);

endmodule
























