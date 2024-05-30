module GEMMR2_tb();

`define L2OUT "/home/nitcloud/project/FPGA/Design/HDL/Tactical-helmet/user/sim/RL/out/L2.txt"
integer w_file;
initial begin
    w_file = $fopen(`L2OUT);
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
        idata <= {16'h3266, 16'h34cd}; // 0.2 | 0.3 /-1.25  /0.47
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

//Instance 
reg                 gvalid;
reg                 ivalid;

wire 	            ovalid_d2;
wire [15:0]	        out_d2;

initial begin
    gvalid  = 1'b0;
    ivalid  = 1'b0;
end

reg [7:0] cnt;

always @(posedge sys_clk or posedge sys_rst) begin
    if(sys_rst)begin
        cnt <= 8'd0;
    end
    else if(ovalid_d2)begin
        if(cnt == 8'd255)
            cnt <= 8'd0;
        else 
            cnt <= cnt + 1'b1;
    end
    else begin
        cnt <= 8'd0;
    end
end

always @(*) begin
    if(ovalid)begin
        gvalid  <= 1'b1;
        ivalid  <= 1'b1;
    end
    else if(cnt == 8'd255)begin
        gvalid  <= 1'b0;
        ivalid  <= 1'b0;
    end
    else begin
        gvalid  <= gvalid;
        ivalid  <= ivalid;
    end
end

GEMMR2 #(
	.K     		( 4  		),
	.WIDTH 		( 16 		))
u_GEMMR2(
	//ports
	.clk    		( sys_clk    	),
	.rst    		( sys_rst    	),

	.gvalid 		( gvalid 		),
	.ivalid 		( ivalid 		),
	.in     		( out     		),

	.ovalid 		( ovalid_d2     ),
	.out    		( out_d2        )
);

reg flag = 0;
always @(posedge sys_clk) begin
    if (ovalid_d2) begin
        flag = 1;
        $fdisplay(w_file, "%b", out_d2);
    end
    else if (!ovalid_d2 && flag) begin
        $finish;
    end
end

endmodule  //TOP
