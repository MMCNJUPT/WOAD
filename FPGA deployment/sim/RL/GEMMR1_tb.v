module GEMMR1_tb();

`define L1OUT "E:/fpga/github/help/hell/L1.txt"
integer w_file;
initial begin
    w_file = $fopen(`L1OUT);
end

parameter MAIN_FRE = 250; //unit MHz
reg       sys_clk  = 0;
reg       sys_rst  = 1; // active high

always begin
    #(500/MAIN_FRE) sys_clk = ~sys_clk;
end

always begin
    #50 sys_rst = 0;
end

reg  trig = 0;
reg  vaild = 0;
reg  vaild_buf = 0;
wire vaild_nege =  ~vaild & vaild_buf;
reg [31:0] idata = 0;
always@(posedge sys_clk) begin
    vaild <= sys_rst;
    vaild_buf <= vaild;
    if (vaild_nege) begin
        trig  <= 1'b1;
        idata <= {16'h3266, 16'h34cd}; // 0.2 | 0.3 /2
    end
    else begin
        trig  <= 0;
        idata <= 0;
    end
end

//Instance 
wire 	    ovalid;
wire [15:0]	out;

GEMMR1 #(
	.LENGTH 		( 256 		),
	.WIDTH  		( 16  		),
	.DEPTH  		( 2   		))
u_GEMMR1(
	//ports
	.clk    		( sys_clk    ),
	.rst    		( sys_rst    ),
	.trig   		( trig       ),
	.in     		( idata      ),
	.ovalid 		( ovalid     ),
	.out    		( out        )
);

reg flag = 0;
always @(posedge sys_clk) begin
    if (ovalid) begin
        flag = 1;
        $fdisplay(w_file, "%b", out);
    end
    else if (!ovalid && flag) begin
        $finish;
    end
end

endmodule  //TOP
