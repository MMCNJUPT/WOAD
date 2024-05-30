module exp_taylor #(
        parameter EXP = 5,
        parameter FRA = 10
    ) (
        input               aclk,
        input               aresetn,

        //S_AXIS
        input [EXP+FRA:0]   s_axis_tdata,
        input               s_axis_tvalid,
        output              s_axis_tready,

        //M_AXIS_RESULT
        output  [EXP+FRA:0] m_axis_result_tdata,
        output              m_axis_result_tvalid,

        output [2:0]       flag
    );

    // e^x = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6!
    localparam taylor_iter = 5;

    //assign mul_out[0] = s_axis_tdata;

    reg [6:0] valid = 7'b0;

    reg [EXP+FRA:0] mul_out[taylor_iter - 2 : 0];

    genvar i;
    generate for(i=0;i<taylor_iter - 6;i=i+1) begin : power
        wire [EXP+FRA:0] mul_buf[taylor_iter - 3 : 0];

        always @(*) begin
            mul_out[0] <= s_axis_tdata;
        end

        always @(posedge aclk) begin
            mul_out[i+1] <= mul_buf[i];
        end

        cmult #(
            .EXP 		( EXP 		),
            .FRA 		( FRA 		))
        u_nmult(
            //ports
            .aresetn    ( aresetn        ),
            .valid      ( valid[i]       ),
            .A    		( s_axis_tdata   ),
            .B    		( mul_out[i]     ),
            .Y    		( mul_buf[i]     ),
            .flag 		(  		      )
        );
        end
    endgenerate

    reg  [15:0] r_mul[3:0];

    always @(posedge aclk) begin
        r_mul[0] <= mul_out[taylor_iter - 6];
        r_mul[1] <= r_mul[0];
        r_mul[2] <= r_mul[1];
        r_mul[3] <= r_mul[2];
    end

    wire  [15:0] div[taylor_iter - 3 : 0];
    wire  [15:0] divide[taylor_iter - 3 : 0];

    assign divide[0] = 16'h3800;
    assign divide[1] = 16'h3155;
    assign divide[2] = 16'h2955;
    assign divide[3] = 16'h2044;
    assign divide[4] = 16'h15b0;
    assign divide[5] = 16'h0a80;
    assign divide[6] = 16'h01a0;

    genvar j;
    generate for(j=0;j<taylor_iter - 6;j=j+1) begin : divide_gen
            cmult #(
                .EXP 		( EXP 		),
                .FRA 		( FRA 		))
            u_div(
                //ports
                .aresetn    ( aresetn        ),
                .valid      ( valid[j]       ),
                .A    		( mul_out[j+1]   ),
                .B    		( divide[j]      ),
                .Y    		( div[j]         ),
                .flag 		(  		         )
            );
        end
    endgenerate

    //1+x
    wire [15:0] add_one;

    cadd #(
        .EXP 		( EXP 		),
        .FRA 		( FRA 		))
    u_add_sub(
        //ports
        .aresetn    ( aresetn           ),
        .valid      ( s_axis_tvalid     ),
        .A 		    ( 16'h3c00 		    ),
        .B 		    ( s_axis_tdata 		),
        .Y 		    ( add_one    		),
        .flag       (                   )
    );//the first

    wire [15:0] add_out[taylor_iter - 2 : 0];

    assign add_out[0] = add_one;

    //x^4/120*x
    genvar m,n;
    generate for(m=0;m<taylor_iter - 5;m=m+1)begin : mul_div
        wire [EXP+FRA:0] mul_buf[taylor_iter - 6 : 0];
        wire [EXP+FRA:0] mul_reg[taylor_iter - 5 : 0];

        assign mul_reg[0] = mul_out[m+4];

        always @(*) begin
            mul_out[m+4] <= mul_buf[m];
        end

        assign div[m+3] = mul_reg[m+1];

        cmult #(
            .EXP 		( EXP 		),
            .FRA 		( FRA 		))
        r_div(
            //ports
            .aresetn    ( aresetn        ),
            .valid      ( valid[m]       ),
            .A    		( r_mul[m]       ),
            .B    		( divide[m+3]    ),
            .Y    		( mul_buf[m]     ),
            .flag 		(  		         )
        );
        for(n=0;n<m+1;n=n+1)begin
            cmult #(
                .EXP 		( EXP 		),
                .FRA 		( FRA 		))
            r_nmult(
            //ports
                .aresetn    ( aresetn        ),
                .valid      ( valid[n]       ),
                .A    		( s_axis_tdata   ),
                .B    		( mul_reg[n]     ),
                .Y    		( mul_reg[n+1]   ),
                .flag 		(  		         )
             );
        end
    end
    endgenerate

    genvar k;
    generate
        for(k=0;k<taylor_iter - 2;k=k+1) begin
            cadd #(
                .EXP 		( EXP 		),
                .FRA 		( FRA 		))
            u_add(
                //ports
                .aresetn    ( aresetn           ),
                .valid      ( valid[k]          ),
                .A 		    ( add_out[k] 		),
                .B 		    ( div[k] 		    ),
                .Y 		    ( add_out[k+1] 		),
                .flag       (                   )
            );
        end
    endgenerate

    cksp #(
        .EXP (EXP),
        .FRA (FRA))
    u_cksp(
        .expo	   (m_axis_result_tdata[EXP+FRA-1:FRA]),
        .frac	   (m_axis_result_tdata[FRA-1:0]),

        .flag 	   (flag)
    );

    always @(posedge aclk) begin
        valid <= {valid[5:0],s_axis_tvalid};
    end

    assign s_axis_tready        = s_axis_tvalid;
    assign m_axis_result_tdata  = add_out[taylor_iter - 2];
    assign m_axis_result_tvalid = valid[6];

endmodule


