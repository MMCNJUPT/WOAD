`include "../../src/driver/uart/uart_tx.v"
module RL_top_tb();

parameter MAIN_FRE   = 100; //unit MHz
reg                   sys_clk = 0;
reg                   sys_rst = 1;

always begin
    #(500/MAIN_FRE) sys_clk = ~sys_clk;
end

always begin
    #50 sys_rst = 0;
end

//Instance 
wire 	ovalid;
wire 	out;
wire 	led;

reg [7:0] uart_din;
reg       uart_en;

initial begin
    uart_en = 1'b0;
    #200
    uart_en = 1'b1;
end

/*
uart_tx u_uart_tx(
	.sys_clk                (sys_clk),
	.uart_tx_data           (uart_tx_data),	      
	.uart_tx_en             (uart_tx_en),			        
	.uart_tx                (uart_tx),
	.uart_tx_done           (uart_tx_done)
);
*/
// outports wire
wire       	uart_txd;

uart_tx u_uart_tx(
    .sys_clk  	( sys_clk   ),
    .sys_rst  	( sys_rst   ),
    
    .uart_en  	( uart_en   ),
    .uart_din 	( uart_din  ),
    
    .uart_txd 	( uart_txd  )
);

reg [2:0] cnt = 3'd0;

always @(posedge sys_clk) begin
    if(uart_tx_done)begin
        if(cnt == 3'd5)begin
            cnt <= 3'd0;
        end
        else begin
            cnt <= cnt + 1'b1;
        end
    end
end

always @(*) begin
    case(cnt)
        3'd0: uart_tx_data <= 8'h4B; //O
        3'd1: uart_tx_data <= 8'h4C; //P
        3'd2: uart_tx_data <= 8'h66; //f
        3'd3: uart_tx_data <= 8'h2e; //.
        3'd4: uart_tx_data <= 8'h66; //f
        3'd5: uart_tx_data <= 8'h32; //2
        default:uart_tx_data <= 8'h00;
    endcase
end

RL_top u_RL_top(
	//ports
	.sys_clk 		( sys_clk 		),
	.sys_rst 		( ~sys_rst 		),
	.uart_rx 		( uart_tx 		),
	.ovalid  		( ovalid  		),
	.out     		( out     		)
);

endmodule  //TOP
