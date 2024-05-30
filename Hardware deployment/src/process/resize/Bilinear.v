`timescale 1ns/1ns
// `include "../../config.v"
`define sim_fun
`define img_output  "/home/project/FPGA/Design/TCL_project/Prj/Tactical-helmet/user/sim/vip/data/output.txt"
module Bilinear #(
    parameter IMG_HDISP_I = 1280,	//1280*720
    parameter IMG_VDISP_I = 720,
    parameter IMG_HDISP_O = 640,	//640*640
    parameter IMG_VDISP_O = 640,
    parameter IMG_DATAW = 16
) (
	//global clock
	input                 pix_clk,  			//cmos video pixel clock
	input                 sys_rstn,				//global reset
	//Image data prepred to be processd
    input                 per_frame_clken,	    //Prepared Image data vsync valid signal
	input                 per_frame_vsyn,	    //Prepared Image data vsync valid signal
	input                 per_frame_href,		//Prepared Image data href vaild  signal
	input [IMG_DATAW-1:0] per_frame_data,		//Prepared Image brightness input

    output [$clog2(IMG_HDISP_O)-1:0]  pix_xaddr,
    output [$clog2(IMG_VDISP_O)-1:0]  pix_yaddr,
    output [IMG_DATAW-1:0]  pix_data
);
integer w_file;
initial begin
    w_file = $fopen(`img_output);
end
// localparam IMG_HDISP_I = 1280;	//1280*720
// localparam IMG_VDISP_I = 720;
// localparam IMG_HDISP_O = 640;	//640*640
// localparam IMG_VDISP_O = 640;

// 首先进行行缓存 state:finished
wire [IMG_DATAW-1:0]    shiftout;

RAMshift_taps #(
	.DELAY_LEN 		( IMG_HDISP_I ),
	.DATA_WIDTH     ( IMG_DATAW   ))
u_RAMshift_taps(
	//ports
	.clock        		( pix_clk        	),
	.clken        		( per_frame_clken   ),
	.shiftin      		( per_frame_data    ), // P1, P2
	.shiftout     		( shiftout     		)  // P3, P4
);

// 检测行同步信号边沿，一行数据的开始和结束
reg  href, vsyn;
reg  href_r, vsyn_r;
wire href_neg =  ~href & href_r;
wire vsyn_neg =  ~vsyn & vsyn_r;
always@(posedge pix_clk or negedge sys_rstn) begin
    if (!sys_rstn) begin
        href <= 0; href_r <= 0;
        vsyn <= 0; vsyn_r <= 0;
    end
    else begin    
        href <= per_frame_href; href_r <= href;
        vsyn <= per_frame_vsyn; vsyn_r <= vsyn;
    end
end

// 获取坐标
reg [$clog2(IMG_HDISP_I)-1:0] x_src_coordinate; //列计数器，横坐标 X
reg [$clog2(IMG_VDISP_I)-1:0] y_src_coordinate; //行计数器，纵坐标 Y
always @ (posedge pix_clk or negedge sys_rstn) begin
	if(!sys_rstn) begin
		x_src_coordinate <= 0;
		y_src_coordinate <= 0;
	end
	else begin
        if (per_frame_clken) begin
            x_src_coordinate <= x_src_coordinate + 1;
        end
        else if (href_neg) begin
            x_src_coordinate <= 0;
            y_src_coordinate <= y_src_coordinate + 1;
        end
        else if (vsyn_neg) begin
            y_src_coordinate <= 0;
        end
    end
end

// control
reg [$clog2(IMG_HDISP_O)-1:0] x_dis_coordinate; //列计数器，横坐标 X
reg [$clog2(IMG_VDISP_O)-1:0] y_dis_coordinate; //行计数器，纵坐标 Y

wire [3:0] dy1 = y_dis_coordinate[2:0]<<1 ^ 15; // 15-2m
wire [3:0] dy2 = y_dis_coordinate[2:0]<<1 | 1;  // 2m+1

// RGB全按6bit来算，系数K为4bit，相加后多1bit，一共11bit
reg [10:0] M1_R,  M1_G,  M1_B;
reg [10:0] M1_BR, M1_BG, M1_BB;
reg [10:0] M2_R,  M2_G,  M2_B;
reg [10:0] DP_R,  DP_G,  DP_B;
// 像素y坐标上计算有效 [(0,1) (1,2) ... (6,7)]有效 [(7,0)]无效 (shiftout,per_frame_data)
// shiftout : P1 P2 ---------> * (dy1=15-2m)
// per_frame_data : P3 P4 ---> * (dy2=2m+1)
reg [3:0] y_cnt;
always @ (posedge pix_clk or negedge sys_rstn) begin
	if(!sys_rstn) begin
        y_cnt <= 0;
	end
	else begin
        if (href_neg) begin
            y_cnt <= y_cnt + 1;
            if (y_cnt == 8) begin
                y_cnt <= 0;
            end
        end
        if (vsyn_neg) begin
            y_cnt <= 0;
        end
    end
end

wire ycal_valid = (y_cnt != 0) & per_frame_clken;
always @ (posedge pix_clk or negedge sys_rstn) begin
	if(!sys_rstn) begin
        x_dis_coordinate <= 0;
        y_dis_coordinate <= 0;
	end
	else begin
        if (ycal_valid && x_src_coordinate[0]) begin // 像素y坐标上计算有效
            // 像素x坐标上计算
            x_dis_coordinate <= x_dis_coordinate + 1;
            if (x_dis_coordinate == IMG_HDISP_O - 1) begin
                x_dis_coordinate <= 0;
                y_dis_coordinate <= y_dis_coordinate + 1;
            end
        end
        if (vsyn_neg) begin
            y_dis_coordinate <= 0;
        end
    end
end

always @ (posedge pix_clk or negedge sys_rstn) begin
	if(!sys_rstn) begin
		M1_R  <= 0; M1_G  <= 0; M1_B  <= 0;
        M2_R  <= 0; M2_G  <= 0; M2_B  <= 0;
        M1_BR <= 0; M1_BG <= 0; M1_BB <= 0;
        DP_R  <= 0; DP_G  <= 0; DP_B  <= 0;
	end
	else begin
        if (ycal_valid) begin // 像素y坐标上计算有效
            // 像素x坐标上计算
            if (x_src_coordinate[0]) begin // [(15-2m)*P2 + (2m+1)*P4] 
                M2_R <= (shiftout[15:11]*dy1 + per_frame_data[15:11]*dy2); // R 5
                M2_G <= (shiftout[10:5] *dy1 + per_frame_data[10:5] *dy2); // G 6
                M2_B <= (shiftout[4:0]  *dy1 + per_frame_data[4:0]  *dy2); // B 5
                DP_R <= (M2_R + M1_BR);
                DP_G <= (M2_G + M1_BG);
                DP_B <= (M2_B + M1_BB);
                // $display("R:%b, G:%b, B:%b", DP_R, DP_G, DP_B);
                $fdisplay(w_file, "%b", pix_data);
                if((x_dis_coordinate == IMG_HDISP_O-1) && (y_dis_coordinate == IMG_VDISP_O-1)) begin
                    #500 $finish;
                end
            end
            else begin // [(15-2m)*P1 + (2m+1)*P3] 
                M1_R <= (shiftout[15:11]*dy1 + per_frame_data[15:11]*dy2); // R
                M1_G <= (shiftout[10:5] *dy1 + per_frame_data[10:5] *dy2); // G
                M1_B <= (shiftout[4:0]  *dy1 + per_frame_data[4:0]  *dy2); // B
                M1_BR <= M1_R; M1_BG <= M1_G; M1_BB <= M1_B;
            end
        end
    end
end

assign pix_data  = {DP_R[9:5], DP_G[10:5], DP_B[9:5]};
assign pix_xaddr = x_dis_coordinate;
assign pix_yaddr = y_dis_coordinate;

endmodule