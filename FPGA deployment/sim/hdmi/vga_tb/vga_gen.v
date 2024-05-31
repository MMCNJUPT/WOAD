//----------------------------------------------------------------------------
// Project Name   : vga_top
// Description    : vga signal generate, with 1280 x 720 pixel.
//----------------------------------------------------------------------------
//  Version             Comments
//------------      ----------------
//    0.1              Created
//----------------------------------------------------------------------------

module vga_gen(
    output [7:0] RED,
    output [7:0] GREEN,
    output [7:0] BLUE,
    output       HSYNC,
    output       VSYNC,
    output       READY,

    input        CLK,  // 74.25 MHz, for 1280 x 720 pixel clock
    input        RST_N
);

wire rdy;
wire [10:0] cl_adr;
wire [10:0] rw_adr;
assign READY = rdy;

sync_gen SYNC_GEN_INST(
    .Hsync(HSYNC),
    .Vsync(VSYNC),
    .ready(rdy),
    .col_addr(cl_adr),
    .row_addr(rw_adr),

    .clk(CLK),
    .rst_n(RST_N)
);

vga_color VGA_COLOR_INST(
    .R(RED),
    .G(GREEN),
    .B(BLUE),

    .clk(CLK),
    .rst_n(RST_N),
    .col_addr(cl_adr),
    .row_addr(rw_adr),
    .ready(rdy)
);

endmodule


