`timescale 1ns / 1ps

module async_fifo_tb();
    
parameter W = 4'd8;

reg		        wr_clk = 0; 
reg		        wr_reset_n; 
reg		        wr_en; 
wire 	        full; 
wire	        afull;
reg  [W-1 : 0]	wr_data = 'd0;

reg		        rd_clk = 0; 
reg		        rd_reset_n;
reg		        rd_en;
wire			empty;
wire			aempty; 
wire [W-1 : 0]	rd_data;

initial begin
    wr_reset_n = 0;
    rd_reset_n = 0;
    wr_en      = 0;
    rd_en      = 0;
    #200
    wr_en      = 1;
    wr_reset_n = 1;
    rd_reset_n = 1;
    #20000
    wr_en      = 0;
    rd_en      = 1;
    #20000
    rd_en      = 0;
    $stop;
end

always #10 wr_clk = ~wr_clk; 
always #30  rd_clk = ~rd_clk; 

always @(posedge wr_clk)begin
    if(wr_en)
        wr_data <= wr_data + 1'b1;
    else 
        wr_data <= wr_data;
end
   
async_fifo #(
   .W   (W)
) 
async_fifo_u(
	//timing for wr
	.wr_clk                (wr_clk), 
	.wr_reset_n            (wr_reset_n), 
	.wr_en                 (wr_en), 
	.full                  (full ), 
	.afull                 (afull), 
	.wr_data               (wr_data),
	
	.rd_clk                (rd_clk), 
	.rd_reset_n            (rd_reset_n),
	.rd_en                 (rd_en),
	.empty                 (empty),
	.aempty                (aempty),      
	.rd_data               (rd_data)
);    
    
endmodule
