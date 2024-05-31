//----------------------------------------------------------------------------
// Project Name   : vga_top
// Description    : vga output signal generate, here to generate 7-colors.
//----------------------------------------------------------------------------
//  Version             Comments
//------------      ----------------
//    0.1              Created
//----------------------------------------------------------------------------

module vga_color(
    output reg [7:0]  R,
    output reg [7:0]  G,
    output reg [7:0]  B,
    
    input             clk,
    input             rst_n,
    input      [10:0] col_addr,
    input      [10:0] row_addr,
    input             ready
);

// to display a color rectangle
// color_table = {rect_red, rect_orange, rect_yellow, rect_green, 
//                rect_blue, rect_violet, rect_black, rect_white};
reg [7:0] color_table = 0;

always @(posedge clk, negedge rst_n) begin
    if(!rst_n)
        color_table <= 8'b0000_0000;
    else if(col_addr >= 11'd0 && col_addr < 11'd1280) begin
        if(row_addr >= 11'd0 && row_addr < 11'd90)
            color_table <= 8'b1000_0000;
        else if(row_addr < 11'd180)
            color_table <= 8'b0100_0000;
        else if(row_addr < 11'd270)
            color_table <= 8'b0010_0000;
        else if(row_addr < 11'd360)
            color_table <= 8'b0001_0000;
        else if(row_addr < 11'd450)
            color_table <= 8'b0000_1000;
        else if(row_addr < 11'd540)
            color_table <= 8'b0000_0100;
        else if(row_addr < 11'd630)
            color_table <= 8'b0000_0010;
        else //if(row_addr < 11'd720)
            color_table <= 8'b0000_0001;
    end
    else
        color_table <= 8'b0000_0000;
end

//----------------------------------------------------------------------------
always @(color_table, ready) begin
    if(ready) begin
        case(color_table) 
            8'b0000_0000: begin R = 8'b0000_0000; G = 8'b0000_0000; B = 8'b0000_0000; end
            8'b1000_0000: begin R = 8'b1111_1111; G = 8'b0000_0000; B = 8'b0000_0000; end
            8'b0100_0000: begin R = 8'b1111_1111; G = 8'b1000_0000; B = 8'b0000_0000; end
            8'b0010_0000: begin R = 8'b1111_1111; G = 8'b1111_1111; B = 8'b0000_0000; end
            8'b0001_0000: begin R = 8'b0000_0000; G = 8'b1111_1111; B = 8'b0000_0000; end
            8'b0000_1000: begin R = 8'b0000_0000; G = 8'b0000_0000; B = 8'b1111_1111; end
            8'b0000_0100: begin R = 8'b1000_0000; G = 8'b0000_0000; B = 8'b1000_0000; end
            8'b0000_0010: begin R = 8'b0000_0000; G = 8'b0000_0000; B = 8'b0000_0000; end
            8'b0000_0001: begin R = 8'b1111_1111; G = 8'b1111_1111; B = 8'b1111_1111; end
            default     : begin R = 8'b0000_0000; G = 8'b0000_0000; B = 8'b0000_0000; end
        endcase
    end
    else begin
        R = 8'b0000_0000;
        G = 8'b0000_0000;
        B = 8'b0000_0000;
    end
end

endmodule


