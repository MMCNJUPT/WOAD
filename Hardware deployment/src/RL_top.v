module RL_top (
    input  sys_clk,
    input  sys_rst,

    input  uart_rx,

    output ovalid,
    output out,
	output uart_tx
);

wire [7:0] uart_rx_data;
wire       uart_rx_done;

uart_rx #(
	.CLK_FREQ (50_000_000),
	.baud     (115200),
	.BAUD_CNT (50_000_000/115200)
)
uart_rx_u(
    .sys_clk			(sys_clk),   //系统时钟
    .sys_rst			(sys_rst),   //系统复位
    .uart_rxd			(uart_rx),  //接收到的串行数据
    
    .uart_done			(uart_rx_done), //串转并完成的标志
    .uart_data			(uart_rx_data)  //转换之后的并行数据
    );

(* mark_debug = "TRUE" *)wire  [31:0] data_out_0;
wire [15:0] data_out_1;
wire [15:0] data_out_2;
(* mark_debug = "TRUE" *)wire 		uart_done;

wire 	   uart_tx_done;

wire [7:0] uart_tx_data;

wire	   tx_sig_q;
wire	   uart_tx_en;

uart_tx uart_tx_u(
    .sys_clk        (sys_clk),   //系统时钟
    .sys_rst        (sys_rst),   //系统复位
    .uart_en        (uart_rx_done),   //上升沿数据开始发送
    .uart_din       (uart_rx_data),  //输入串行数据
    
    .uart_txd       (uart_tx)  //转换之后的并行数据
    ); 

uart_rx_control u_uart_rx_control(
	//ports
	.clk_50m      		( sys_clk      		),
	.rst_n        		( sys_rst        	),
	.uart_rx_done 		( uart_rx_done 		),
	.uart_rx_data 		( uart_rx_data 		),
	.data_out_0   		( data_out_0   		),
	//.data_out_1   		( data_out_1   		),
	//.data_out_2   		( data_out_2   		),
	.uart_done    		( uart_done    		)
);

RL #(
	.LENGTH     (256),
    .WIDTH      (16),
    .DEPTH      (2)
)
u_RL(
    // there is ...
    .sys_clk            (sys_clk), // this is system clock
    .sys_rst            (~sys_rst),

    .trig               (uart_done),
    .in                 (data_out_0),

    .ovalid             (ovalid),
    .out                (out)
);

endmodule

