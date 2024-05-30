module RL #(
	parameter    LENGTH = 256,
    parameter    WIDTH = 16,
    parameter    DEPTH = 2
)(
    // there is ...
    input        			sys_clk, // this is system clock
    input        			sys_rst,
    
    input        			trig,
    input [WIDTH*DEPTH-1:0] in,

    output       			ovalid,
    output       			out
);

//Instance 
wire 	    ovalid_d1;
wire [15:0]	out_d1;

GEMMR1 #(
	.LENGTH 		( 256 		),
	.WIDTH  		( 16  		),
	.DEPTH  		( 2   		))
u_GEMMR1(
	//ports
	.clk    		( sys_clk    ),
	.rst    		( sys_rst    ),
	.trig   		( trig       ),
	.in     		( in      	 ),
	.ovalid 		( ovalid_d1  ),
	.out    		( out_d1     )
);

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
    if(ovalid_d1)begin
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
	.in     		( out_d1     	),

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

(* mark_debug = "TRUE" *)reg [15:0] others_a;
(* mark_debug = "TRUE" *)reg [15:0] others_b;

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

assign ovalid = others_valid;
assign out    = index;

endmodule //RL
