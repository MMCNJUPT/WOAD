`timescale 1ns / 1ps

module uart_tx(
    input            sys_clk,   //ϵͳʱ��
    input            sys_rst,   //ϵͳ��λ
    input            uart_en,   //���������ݿ�ʼ����
    input [7:0]      uart_din,  //���봮������
    
    output reg       uart_txd  //ת��֮��Ĳ�������
    );
    
parameter CLK_FREQ = 50_000_000;
parameter baud     = 115200;
localparam BAUD_CNT = CLK_FREQ/baud; //����������
 
reg        uart_en_d0;
reg        uart_en_d1;  //���ڰ��첽����ͬ����ʱ���£���ֹ��������̬
wire       en_flag;   //��uart_rxd�½���ʱ���ݿ�ʼ����
reg        tx_flag;      //�����ݴ������������
reg [15:0] baud_cnt;     //����������
reg [3:0]  tx_cnt;       //����ʱ��
reg [7:0]  tx_data;      //��������

//��ʱһ�����ڵ��Ǹߵ�ƽ����ʱ�������ڵ��ǵ͵�ƽ
assign en_flag = (~uart_en_d1) & uart_en_d0;
  
//�첽����ͬ����ʱ����
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)begin
       uart_en_d0 <= 1'b0;
       uart_en_d1 <= 1'b0;  
    end
    else begin
       uart_en_d0 <= uart_en;     //��ʱһ������
       uart_en_d1 <= uart_en_d0;  //��ʱ��������
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

//��������תΪ��������
always @(posedge sys_clk or negedge sys_rst)begin
    if(!sys_rst)
        uart_txd <= 1'b1;  //����ʱ���ڷ��Ͷ�Ϊ�ߵ�ƽ
    else if(tx_flag)begin
        case(tx_cnt)
            4'd0:uart_txd <= 1'b0; //��ʼλΪ�͵�ƽ
            4'd1:uart_txd <= tx_data[0];
            4'd2:uart_txd <= tx_data[1];
            4'd3:uart_txd <= tx_data[2];
            4'd4:uart_txd <= tx_data[3];
            4'd5:uart_txd <= tx_data[4];
            4'd6:uart_txd <= tx_data[5];
            4'd7:uart_txd <= tx_data[6];
            4'd8:uart_txd <= tx_data[7];
            4'd9:uart_txd <= 1'b1; //ֹͣλΪ�ߵ�ƽ
            default:;
         endcase
    end
    else
        uart_txd <= 1'b1; //����λ�Ǹߵ�ƽ
end   
    
endmodule
