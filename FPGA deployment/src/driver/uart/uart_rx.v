`timescale 1ns / 1ps

module uart_rx(
    input            sys_clk,   //ϵͳʱ��
    input            sys_rst,   //ϵͳ��λ
    input            uart_rxd,  //���յ��Ĵ�������
    
    output reg       uart_done, //��ת����ɵı�־
    output reg [7:0] uart_data  //ת��֮��Ĳ�������
    );
    
parameter CLK_FREQ = 50_000_000;
parameter baud     = 115200;
parameter BAUD_CNT = CLK_FREQ/baud; //����������
 
reg [7:0]  uart_rxd_d0;
reg [7:0]  uart_rxd_d1;  //���ڰ��첽����ͬ����ʱ���£���ֹ��������̬
wire       start_flag;   //��uart_rxd�½���ʱ���ݿ�ʼ����
reg        rx_flag;      //�����ݴ������������
reg [15:0] baud_cnt;     //����������
reg [3:0]  rx_cnt;       //����ʱ��
reg [7:0]  rx_data;      //��������

//��ʱһ�����ڵ��ǵ͵�ƽ����ʱ�������ڵ��Ǹߵ�ƽ
assign start_flag = uart_rxd_d1 & (~uart_rxd_d0);
  
//�첽����ͬ����ʱ����
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)begin
       uart_rxd_d0 <= 1'b0;
       uart_rxd_d1 <= 1'b0;  
    end
    else begin
       uart_rxd_d0 <= uart_rxd;     //��ʱһ������
       uart_rxd_d1 <= uart_rxd_d0;  //��ʱ��������
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

//�������
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
