module fdiv_tb();

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
wire 	input_a_ack;
wire 	input_b_ack;
wire [31:0]	output_z;
wire 	output_z_stb;

reg  [31:0] input_a;
reg     	input_a_stb;
reg  [31:0] input_b;
reg 		input_b_stb;
reg     	output_z_ack;

initial begin
	input_a_stb  = 1'b1;
	input_b_stb  = 1'b1;
	output_z_ack = 1'b1;
	input_a		 = 32'h000116C2;
	input_b		 = 32'h3DCCCCCD;
end

divider #(
	.get_a         		( 4'd0  		),
	.get_b         		( 4'd1  		),
	.unpack        		( 4'd2  		),
	.special_cases 		( 4'd3  		),
	.normalise_a   		( 4'd4  		),
	.normalise_b   		( 4'd5  		),
	.divide_0      		( 4'd6  		),
	.divide_1      		( 4'd7  		),
	.divide_2      		( 4'd8  		),
	.divide_3      		( 4'd9  		),
	.normalise_1   		( 4'd10 		),
	.normalise_2   		( 4'd11 		),
	.round         		( 4'd12 		),
	.pack          		( 4'd13 		),
	.put_z         		( 4'd14 		))
u_divider(
	//ports
	.clk          		( sys_clk          	),
	.rst          		( sys_rst           ),
	.input_a      		( input_a      		),
	.input_a_stb  		( input_a_stb  		),
	.input_a_ack  		( input_a_ack  		),
	.input_b      		( input_b      		),
	.input_b_stb  		( input_b_stb  		),
	.input_b_ack  		( input_b_ack  		),
	.output_z     		( output_z     		),
	.output_z_stb 		( output_z_stb 		),
	.output_z_ack 		( output_z_ack 		)
);



endmodule  //TOP
