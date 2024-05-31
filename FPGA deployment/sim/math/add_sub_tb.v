module add_sub_tb();

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
reg                 valid;
reg     [EXP+FRA:0] A;
reg     [EXP+FRA:0] B;
wire    [EXP+FRA:0]	Y;
wire    [2:0]       flag;

initial begin
    A     = 16'h3c6b;
    B     = 16'h0974;
    valid = 1'b0;
    #100
    valid = 1'b1;
end

cadd #(
	.EXP 		( EXP 		),
	.FRA 		( FRA 		))
u_add_sub(
	//ports
    .aresetn ( sys_rst  ),
    .valid   ( valid    ),
	.A 		 ( A 		),
	.B 		 ( B 		),
	.Y 		 ( Y 		),
    .flag    ( flag     )
);

initial begin            
    $dumpfile("wave.vcd");        
    $dumpvars(0, add_sub_tb);    
    #50000 $finish;
end

endmodule  //TOP
