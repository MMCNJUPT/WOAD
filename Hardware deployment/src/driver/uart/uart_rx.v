`timescale 1ns / 1ps

module uart_rx(
    input            sys_clk,   //系统时钟
    input            sys_rst,   //系统复位
    input            uart_rxd,  //接收到的串行数据
    
    output reg       uart_done, //串转并完成的标志
    output reg [7:0] uart_data  //转换之后的并行数据
    );
    
parameter CLK_FREQ = 50_000_000;
parameter baud     = 115200;
parameter BAUD_CNT = CLK_FREQ/baud; //波特率周期
 
reg [7:0]  uart_rxd_d0;
reg [7:0]  uart_rxd_d1;  //用于把异步数据同步到时钟下，防止产生亚稳态
wire       start_flag;   //在uart_rxd下降沿时数据开始传输
reg        rx_flag;      //在数据传输过程中拉高
reg [15:0] baud_cnt;     //波特率周期
reg [3:0]  rx_cnt;       //接收时间
reg [7:0]  rx_data;      //接收数据

//延时一个周期的是低电平，延时两个周期的是高电平
assign start_flag = uart_rxd_d1 & (~uart_rxd_d0);
  
//异步数据同步到时钟下
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)begin
       uart_rxd_d0 <= 1'b0;
       uart_rxd_d1 <= 1'b0;  
    end
    else begin
       uart_rxd_d0 <= uart_rxd;     //延时一个周期
       uart_rxd_d1 <= uart_rxd_d0;  //延时两个周期
    end
end   
    
//rx_flag
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)
        rx_flag <= 1'b0;
    else if(start_flag == 1'b1)
        rx_flag <= 1'b1;
    else if(rx_cnt == 4'd9&&(baud_cnt == BAUD_CNT/2-1'b1))
        rx_flag <= 1'b0;
    else
        rx_flag <= rx_flag;
end    

//baud_cnt
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)
        baud_cnt <= 16'd0;
    else if(rx_flag)begin
        if(baud_cnt < BAUD_CNT-1'b1)
            baud_cnt <= baud_cnt + 1'b1;
        else
            baud_cnt <= 16'd0;
    end
    else
        baud_cnt <= 16'd0;
end

//rx_cnt 
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)
        rx_cnt <= 4'd0;
    else if(rx_flag)begin
        if(baud_cnt == BAUD_CNT - 1'b1)
            rx_cnt <= rx_cnt+1'b1;
        else
            rx_cnt <= rx_cnt;
    end
    else
        rx_cnt <= 4'd0;
end

//rx_data
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)
        rx_data <= 8'd0;
    else if(rx_flag)begin
        if(baud_cnt == BAUD_CNT/2)
            case(rx_cnt)
                4'd1:rx_data[0] <= uart_rxd_d1;
                4'd2:rx_data[1] <= uart_rxd_d1;
                4'd3:rx_data[2] <= uart_rxd_d1;
                4'd4:rx_data[3] <= uart_rxd_d1;
                4'd5:rx_data[4] <= uart_rxd_d1;
                4'd6:rx_data[5] <= uart_rxd_d1;
                4'd7:rx_data[6] <= uart_rxd_d1;
                4'd8:rx_data[7] <= uart_rxd_d1;
                default: ;
            endcase
        else
            rx_data <= rx_data;    
    end
    else
        rx_data <= 8'd0;
end 

//传输结束
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)begin
        uart_done <= 1'b0;
        uart_data <= 8'd0;
    end
    else if(rx_cnt == 4'd9)begin
        uart_done <= 1'b1;
        uart_data <= rx_data;
    end
    else begin
        uart_done <= 1'b0;
        uart_data <= 8'd0;
    end
end    
        
endmodule
