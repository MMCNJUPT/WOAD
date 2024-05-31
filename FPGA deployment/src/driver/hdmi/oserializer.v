module oserializer #(
    parameter VIDEO_RATE = 25
) (
    input       clk_pixel,
    input       clk_pixel_x5,
    input       reset,
    input [9:0] red,
    input [9:0] green,
    input [9:0] blue,

    output [2:0] tmds,
    output       tmds_clock
);

parallel_to_serial serial_red(
    .clk1x          (clk_pixel),
    .clk5x          (clk_pixel_x5),
    .rst            (reset),
    .din            (red),
    .dout           (tmds[2])
);

parallel_to_serial serial_green(
    .clk1x          (clk_pixel),
    .clk5x          (clk_pixel_x5),
    .rst            (reset),
    .din            (green),
    .dout           (tmds[1])
);

parallel_to_serial serial_blue(
    .clk1x          (clk_pixel),
    .clk5x          (clk_pixel_x5),
    .rst            (reset),
    .din            (blue),
    .dout           (tmds[0])
);

wire [9:0] clk_10bit;

assign clk_10bit = 10'b11111_00000;

parallel_to_serial serial_clk(
    .clk1x          (clk_pixel),
    .clk5x          (clk_pixel_x5),
    .rst            (reset),
    .din            (clk_10bit),
    .dout           (tmds_clock)
);

endmodule




































