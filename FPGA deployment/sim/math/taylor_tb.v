module taylor_tb();

parameter EXP = 5;
parameter FRA = 11 - 1;
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
reg     [EXP+FRA:0] s_axis_tdata;
reg					s_axis_tvalid;

wire 	            s_axis_tready;
wire [EXP+FRA:0]	m_axis_result_tdata;
wire 	            m_axis_result_tvalid;
wire [2:0]	        flag;

initial begin
	s_axis_tdata  = 16'h2e66;
	s_axis_tvalid = 1'b0;
    #100
    s_axis_tvalid = 1'b1;
end

exp_taylor #(
	.EXP 		( EXP 		),
	.FRA 		( FRA 		))
u_exp_taylor(
	//ports
	.aclk                 		( sys_clk                 	),
	.aresetn              		( sys_rst              		),
	
    .s_axis_tdata         		( s_axis_tdata         		),
	.s_axis_tvalid        		( s_axis_tvalid        		),
	.s_axis_tready        		( s_axis_tready        		),
	
    .m_axis_result_tdata  		( m_axis_result_tdata  		),
	.m_axis_result_tvalid 		( m_axis_result_tvalid 		),
	.flag                 		( flag                 		)
);

initial begin            
    $dumpfile("wave.vcd");        
    $dumpvars(0, taylor_tb);    
    #50000 $finish;
end

endmodule  //TOP
