`timescale 1ns/1ns
// `include "../../src/config.v"
`define sim_fun
`ifdef sim_fun
    `define img_input  "/home/project/FPGA/Design/TCL_project/Prj/Tactical-helmet/user/sim/vip/data/input.txt"
`endif
module CMOS_OUT_tb #(
        parameter CMOS_VSYNC_VALID = 1'b1, //H : Data Valid; L : Frame Sync(Set it by register)
        parameter IMG_HDISP        = 640,  //640*480
        parameter IMG_VDISP        = 480
    ) (
        //global reset
        input              rst_n,

        //CMOS Camera interface and data output simulation
        input              cmos_xclk,            //cmos driver clock
        output             cmos_pclk,            //24MHz when rgb output, 12MHz when raw output
        output             cmos_vsync,            //L: vaild, H: invalid
        output reg         cmos_href,            //H: vaild, L: invalid
        output reg [7:0]   cmos_data            //8 bits cmos data input
    );
    wire clk = cmos_xclk;
    //------------------------------------------
    //generate cmos timing
    /*
    localparam V_SYNC = 11'd3;
    localparam V_BACK = 11'd17;
    localparam V_DISP = IMG_VDISP;    //11'd480
    localparam V_FRONT = 11'd10;
    localparam V_TOTAL = V_SYNC + V_BACK + V_DISP + V_FRONT;    //10'd510

    localparam H_SYNC = 11'd80;
    localparam H_BACK = 11'd45;
    localparam H_DISP = IMG_HDISP;    //11'd640
    localparam H_FRONT = 11'd19;
    localparam H_TOTAL = H_SYNC + H_BACK + H_DISP + H_FRONT;    //10'd784
    */
    //Just for simulation
    localparam H_SYNC  = 11'd3;
    localparam H_BACK  = 11'd2;
    localparam H_DISP  = IMG_HDISP*2;
    localparam H_FRONT = 11'd6;
    localparam H_TOTAL = H_SYNC + H_BACK + H_DISP + H_FRONT;    //10'd784
    localparam H_FIXED = H_SYNC + H_BACK;

    localparam V_SYNC  = 11'd2;
    localparam V_BACK  = 11'd1;
    localparam V_DISP  = IMG_VDISP;
    localparam V_FRONT = 11'd0;
    localparam V_TOTAL = V_SYNC + V_BACK + V_DISP + V_FRONT;    //10'd510
    localparam V_FIXED = V_SYNC + V_BACK;

    localparam F_TOTAL = H_DISP*V_DISP;

    reg pixel_cnt;
    reg [$clog2(H_TOTAL)-1:0] hcnt;
    reg [$clog2(V_TOTAL)-1:0] vcnt;

    reg  [7:0] image [F_TOTAL-1:0];
    wire [$clog2(F_TOTAL)-1:0] addr;
    assign addr = cmos_vsync ? ((vcnt - V_FIXED)*H_DISP + (hcnt - H_FIXED)) : 0;

    //----------------------------------
    ////25MHz when rgb output, 12.5MHz when raw output
    always@(posedge clk or negedge rst_n) begin
        if(!rst_n)
            pixel_cnt <= 0;
        else
            pixel_cnt <= pixel_cnt + 1'b1;
    end
    wire pixel_flag = 1'b1;
    assign cmos_pclk = ~clk;

    //---------------------------------------------
    //Horizontal counter
    always@(posedge clk or negedge rst_n) begin
        if(!rst_n)
            hcnt <= 11'd0;
        else if(pixel_flag)
            hcnt <= (hcnt < H_TOTAL - 1'b1) ? hcnt + 1'b1 : 11'd0;
        else
            hcnt <= hcnt;
    end

    //---------------------------------------------
    //Vertical counter
    always@(posedge clk or negedge rst_n) begin
        if(!rst_n)
            vcnt <= 11'd0;
        else if(pixel_flag) begin
            if(hcnt == H_TOTAL - 1'b1)
                vcnt <= (vcnt < V_TOTAL - 1'b1) ? vcnt + 1'b1 : 11'd0;
            else
                vcnt <= vcnt;
        end
        else
            vcnt <= vcnt;
    end

    //---------------------------------------------
    //Image data vsync valid signal
    reg    cmos_vsync_r;
    always@(posedge clk or negedge rst_n) begin
        if(!rst_n)
            cmos_vsync_r <= 1'b0;         //H: Vaild, L: inVaild
        else if(pixel_flag) begin
            if(vcnt <= V_SYNC - 1'b1)
                cmos_vsync_r <= 1'b0;     //H: Vaild, L: inVaild
            else
                cmos_vsync_r <= 1'b1;     //H: Vaild, L: inVaild
        end
        else
            cmos_vsync_r <= cmos_vsync_r;
    end
    assign    cmos_vsync    =    (CMOS_VSYNC_VALID    == 1'b0) ? ~cmos_vsync_r :    cmos_vsync_r;


    //---------------------------------------------
    //Image data href vaild  signal
    wire frame_valid_ahead = ((vcnt >= V_SYNC + V_BACK  && vcnt < V_SYNC + V_BACK + V_DISP &&
                             hcnt >= H_SYNC + H_BACK  && hcnt < H_SYNC + H_BACK + H_DISP))
                             ? 1'b1 : 1'b0;
    always@(posedge clk or negedge rst_n) begin
        if(!rst_n)
            cmos_href <= 0;
        else if(pixel_flag) begin
            if(frame_valid_ahead)
                cmos_href <= 1;
            else
                cmos_href <= 0;
        end
        else
            cmos_href <= cmos_href;
    end

    //---------------------------------------------
    //CMOS Camera data output
    `ifdef img_input
        initial begin
            $readmemb(`img_input, image);
        end
    `else
        integer i;
        initial begin
           for (i = 0; i<F_TOTAL; i=i+1) begin
                image[i] = i;
            end
        end
    `endif
    always@(posedge clk or negedge rst_n) begin
        if(!rst_n)
            cmos_data <= 16'd0;
        else if(pixel_flag) begin
            if(frame_valid_ahead)
                cmos_data <= image[addr];
            else
                cmos_data <= 0;
        end
        else
            cmos_data <= cmos_data;
    end

endmodule

