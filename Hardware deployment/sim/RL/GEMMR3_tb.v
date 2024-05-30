module GEMMR3_tb();

`define L3OUT "/home/nitcloud/project/FPGA/Design/HDL/Tactical-helmet/user/sim/RL/out/L3.txt"
`define OthersOUT "/home/nitcloud/project/FPGA/Design/HDL/Tactical-helmet/user/sim/RL/out/others.txt"
integer w_file;
integer s_file;

initial begin
    w_file = $fopen(`L3OUT);
end

initial begin
    s_file = $fopen(`OthersOUT);
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
        idata <= {16'h3c00, 16'h4000}; // 0.2 | 0.3 /4 /0.2 //
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

reg r_gvalid;
reg r_ivalid;

wire 	            ovalid_d3;
wire [15:0]	        out_d3;

initial begin
    r_gvalid  = 1'b0;
    r_ivalid  = 1'b0;
end

reg [7:0] r_cnt;

always @(posedge sys_clk or posedge sys_rst) begin
    if(sys_rst)begin
        r_cnt <= 8'd0;
    end
    else if(ovalid_d3)begin
        if(cnt == 8'd1)
            r_cnt <= 8'd0;
        else 
            r_cnt <= r_cnt + 1'b1;
    end
    else begin
        r_cnt <= 8'd0;
    end
end

always @(*) begin
    if(ovalid_d2)begin
        r_gvalid  <= 1'b1;
        r_ivalid  <= 1'b1;
    end
    else if(r_cnt == 8'd1)begin
        r_gvalid  <= 1'b0;
        r_ivalid  <= 1'b0;
    end
    else begin
        r_gvalid  <= r_gvalid;
        r_ivalid  <= r_ivalid;
    end
end

GEMMR3 #(
	.K     		( 4  		),
	.WIDTH 		( 16 		))
u_GEMMR3(
	//ports
	.clk    		( sys_clk    	),
	.rst    		( sys_rst    	),

	.gvalid 		( r_gvalid 		),
	.ivalid 		( r_ivalid 		),
	.in     		( out_d2     	),

	.ovalid 		( ovalid_d3     ),
	.out    		( out_d3        )
);

reg  r_ovalid_d3;
wire ovalid_pos;
wire ovalid_hold;
wire ovalid_neg;

always @(posedge sys_clk) begin
    r_ovalid_d3 <= ovalid_d3;
end

assign ovalid_pos  = (~r_ovalid_d3) && ovalid_d3;
assign ovalid_hold =    ovalid_d3   && r_ovalid_d3;
assign ovalid_neg  = (~ovalid_d3)   && r_ovalid_d3;

reg [15:0] others_a;
reg [15:0] others_b;

always @(*) begin
    if(sys_rst)begin
        others_a <= 1'b0;
        others_b <= 1'b0;
    end
    else if(ovalid_pos)begin
        others_a <= out_d3;
        others_b <= others_b;
    end
    else if(ovalid_hold)begin
        others_a <= others_a;
        others_b <= out_d3;
    end
    else begin
        others_a <= others_a;
        others_b <= others_b;
    end
end

reg others_ivalid;

wire others_valid;
wire others_a_tready;
wire others_b_tready;
wire index;
wire equal;

always @(*) begin
    if(sys_rst)
        others_ivalid <= 1'b0;
    else if(ovalid_neg)
        others_ivalid <= 1'b1;
    else 
        others_ivalid <= others_ivalid;
end

others #(
	.EXP 		( 5  		),
	.FRA 		( 10 		))
u_others(
	//ports
	.aclk                 		( sys_clk                 	),
	.aresetn              		( sys_rst              		),

	.s_axis_a_tdata       		( others_a             		),
	.s_axis_a_tvalid      		( others_ivalid      		),
	.s_axis_a_tready      		( others_a_tready      		),

	.s_axis_b_tdata       		( others_b       		    ),
	.s_axis_b_tvalid      		( others_ivalid      		),
	.s_axis_b_tready      		( others_b_tready      		),

	.m_axis_result_tvalid 		( others_valid 		        ),
	.index                		( index                		),
	.equal                		( equal                		)
);

reg mark = 0;

always @(posedge sys_clk) begin
    if (ovalid_d3) begin
        mark = 1;
        $fdisplay(w_file, "%b", out_d3);
    end
    else if (!ovalid_d3 && mark) begin
        $finish(w_file);
    end
end

wire [1:0] final = {index,equal};

reg flag = 0;
always @(posedge sys_clk) begin
    if (others_valid && (!flag)) begin
        flag = 1;
        $fdisplay(s_file, "%b", final);
    end
    else if(flag)begin
        $finish(s_file);
    end
end

endmodule  //TOP
