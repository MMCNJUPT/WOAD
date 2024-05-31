module sys_config(
    input   iclk,
    input   RST,
    output  clk_pix,
    output  clk_pix_x5,
    output  locked
);

reg rst_s1, rst_s2;

always @ (posedge iclk or negedge RST) begin
    if (RST)begin 
        rst_s1 <= 1'b0;
        rst_s2 <= 1'b0;
    end
    else begin
        rst_s1 <= 1'b1;
        rst_s2 <= rst_s1;
    end
end
assign rst_n = rst_s2; 

wire 	clk_out3;
wire 	clk_out4;

CLK_Global #(
	.Mult         		( 59.375     	),
	.DIVCLK_DIV   		( 4      		),
	.CLKIN_PERIOD 		( 20.000 		),

	.CLKOUT0_DIV  		( 10      		),
	.CLK0_PHASE   		( 0.0    		),

	.CLKOUT1_DIV  		( 2      		),
	.CLK1_PHASE   		( 0.0    		),

	.CLKOUT2_DIV  		( 20      		),
	.CLK2_PHASE   		( 0.0    		),

	.CLKOUT3_DIV  		( 20     		),
	.CLK3_PHASE   		( 0.0    		))
u_CLK_Global(
	//ports
	.iclk   		( iclk   		),
	.RST    		( RST    		),
	.clk_out1 		( clk_pix 	    ),
	.clk_out2 		( clk_pix_x5    ),
	.clk_out3 		( clk_out3 		),
	.clk_out4 		( clk_out4 		),
	.locked   		( locked   		)
);


endmodule  //sys_config