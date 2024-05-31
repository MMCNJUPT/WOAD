//----------------------------------------------------------------------------
// Project Name   : vga_gen
// Description    : Generate Hsync, Vsync, and related signal for vga display.
//----------------------------------------------------------------------------
//
//  For 1280 x 720 @ 60Hz,
//
//      Pixels Clock  = 74.25 Mhz
//
//      H Sync Time   =   40 pixels
//      H Back Porch  =  220 pixels  
//      H Active Time = 1280 pixels
//      H Front Porch =  110 pixels
//
//      V Sync Time   =    5 lines
//      V Back Porch  =   20 lines
//      V Active Time =  720 lines
//      V Front Porch =    5 lines
//
//----------------------------------------------------------------------------
//  Version             Comments
//------------      ----------------
//    0.1              Created
//----------------------------------------------------------------------------

module sync_gen(
    output        Hsync,
    output        Vsync,
    output        ready,
    output [10:0] col_addr,
    output [10:0] row_addr,
    
    input         clk,
    input         rst_n
);

reg [10:0] cnt_H = 0;
always @(posedge clk, negedge rst_n) begin
    if(!rst_n)
        cnt_H <= 11'd0;
    else if(cnt_H == 11'd1649)  // 40 + 220 + 1280 + 110 = 1650
        cnt_H <= 11'd0;
    else
        cnt_H <= cnt_H + 1'b1;
end

reg [10:0] cnt_V = 0;
always @(posedge clk, negedge rst_n) begin
    if(!rst_n)
        cnt_V <= 11'd0;
    else if(cnt_V == 11'd749)  // 5 + 20 + 720 + 5 = 750
        cnt_V <= 12'd0;
    else if(cnt_H == 11'd1649)
        cnt_V <= cnt_V + 1'b1;
end

reg ready_r = 0;
always @(posedge clk, negedge rst_n) begin
    if(!rst_n)
        ready_r <= 1'b0;
    else if( (cnt_H >= 11'd260 && cnt_H < 11'd1540)  // (40+220, 40+220+1280)
        && (cnt_V >= 11'd25 && cnt_V < 11'd745) )   // (5+20, 5+20+720)
        ready_r <= 1'b1;
    else
        ready_r <= 1'b0;
end

//----------------------------------------------------------------------------
assign Hsync = (cnt_H < 11'd40) ? 1'b0 : 1'b1;
assign Vsync = (cnt_V < 11'd5)   ? 1'b0 : 1'b1;
assign ready = ready_r;
assign col_addr = ready_r ? cnt_H - 11'd260 : 11'd0;
assign row_addr = ready_r ? cnt_V - 11'd25  : 11'd0;

endmodule


