module others_tb();

parameter EXP = 5;
parameter FRA = 10;
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
reg [EXP+FRA:0] s_axis_a_tdata;
reg             s_axis_a_tvalid;
wire            s_axis_a_tready;

//S_AXIS_B
reg [EXP+FRA:0] s_axis_b_tdata;
reg             s_axis_b_tvalid;
wire            s_axis_b_tready;

wire 			m_axis_result_tvalid;
wire 			index;
wire 			equal;

initial begin
        s_axis_a_tdata  = 16'hb4ea;
        s_axis_b_tdata  = 16'h3296;
        s_axis_a_tvalid = 1'b0;
        s_axis_b_tvalid = 1'b0;
		#100
		s_axis_a_tvalid = 1'b1;
        s_axis_b_tvalid = 1'b1;
end

others #(
	.EXP 		( EXP  		),
	.FRA 		( FRA 		))
u_others(
	//ports
	.aclk                 		( sys_clk                 	),
	.aresetn              		( sys_rst              		),

	.s_axis_a_tdata       		( s_axis_a_tdata       		),
	.s_axis_a_tvalid      		( s_axis_a_tvalid      		),
	.s_axis_a_tready      		( s_axis_a_tready      		),

	.s_axis_b_tdata       		( s_axis_b_tdata       		),
	.s_axis_b_tvalid      		( s_axis_b_tvalid      		),
	.s_axis_b_tready      		( s_axis_b_tready      		),

	.m_axis_result_tvalid 		( m_axis_result_tvalid 		),
	.index                		( index                		),
	.equal                		( equal                		)
);

initial begin            
    $dumpfile("wave.vcd");        
    $dumpvars(0, others_tb);    
    #50000 $finish;
end

endmodule  //TOP
