`timescale 1ns / 1ps

module uart_tx(
    input            sys_clk,   //系统时钟
    input            sys_rst,   //系统复位
    input            uart_en,   //上升沿数据开始发送
    input [7:0]      uart_din,  //输入串行数据
    
    output reg       uart_txd  //转换之后的并行数据
    );
    
parameter CLK_FREQ = 50_000_000;
parameter baud     = 115200;
localparam BAUD_CNT = CLK_FREQ/baud; //波特率周期
 
reg        uart_en_d0;
reg        uart_en_d1;  //用于把异步数据同步到时钟下，防止产生亚稳态
wire       en_flag;   //在uart_rxd下降沿时数据开始传输
reg        tx_flag;      //在数据传输过程中拉高
reg [15:0] baud_cnt;     //波特率周期
reg [3:0]  tx_cnt;       //接收时间
reg [7:0]  tx_data;      //接收数据

//延时一个周期的是高电平，延时两个周期的是低电平
assign en_flag = (~uart_en_d1) & uart_en_d0;
  
//异步数据同步到时钟下
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)begin
       uart_en_d0 <= 1'b0;
       uart_en_d1 <= 1'b0;  
    end
    else begin
       uart_en_d0 <= uart_en;     //延时一个周期
       uart_en_d1 <= uart_en_d0;  //延时两个周期
    end
end   
    
//tx_flag
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)
        tx_flag <= 1'b0;
    else if(en_flag)
        tx_flag <= 1'b1;
    else if(tx_cnt == 4'd9&&(baud_cnt == BAUD_CNT/2-1'b1))
        tx_flag <= 1'b0;
    else
        tx_flag <= tx_flag;
end    

//baud_cnt
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)
        baud_cnt <= 16'd0;
    else if(tx_flag)begin
        if(baud_cnt < BAUD_CNT-1'b1)
            baud_cnt <= baud_cnt + 1'b1;
        else
            baud_cnt <= 16'd0;
    end
    else
        baud_cnt <= 16'd0;
end

//tx_cnt 
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)
        tx_cnt <= 4'd0;
    else if(tx_flag)begin
        if(baud_cnt == BAUD_CNT - 1'b1)
            tx_cnt <= tx_cnt+1'b1;
        else
            tx_cnt <= tx_cnt;
    end
    else
        tx_cnt <= 4'd0;
end

//tx_data
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)
        tx_data <= 8'd0;
    else if(en_flag)
        tx_data <= uart_din;
    else
        tx_data <= tx_data;
end 

//并行数据转为串行数据
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)
        uart_txd <= 1'b1;  //空闲时串口发送端为高电平
    else if(tx_flag)begin
        case(tx_cnt)
            4'd0:uart_txd <= 1'b0; //起始位为低电平
            4'd1:uart_txd <= tx_data[0];
            4'd2:uart_txd <= tx_data[1];
            4'd3:uart_txd <= tx_data[2];
            4'd4:uart_txd <= tx_data[3];
            4'd5:uart_txd <= tx_data[4];
            4'd6:uart_txd <= tx_data[5];
            4'd7:uart_txd <= tx_data[6];
            4'd8:uart_txd <= tx_data[7];
            4'd9:uart_txd <= 1'b1; //停止位为高电平
            default:;
         endcase
    end
    else
        uart_txd <= 1'b1; //空闲位是高电平
end   
    
endmodule
