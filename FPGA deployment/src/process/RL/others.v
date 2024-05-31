module others #(
        parameter EXP = 5,
        parameter FRA = 10
    ) (
        //Global
        input             aclk,
        input             aresetn,

        //S_AXIS_A
        input [EXP+FRA:0] s_axis_a_tdata,
        input             s_axis_a_tvalid,
        output            s_axis_a_tready,

        //S_AXIS_B
        input [EXP+FRA:0] s_axis_b_tdata,
        input             s_axis_b_tvalid,
        output            s_axis_b_tready,

        //M_AXIS_RESULT
        output            m_axis_result_tvalid,
        output reg        index, //1'b1 : d1 > d2  1'b0 : d1 < d2  1'bz: d1 = d2
        output reg        equal //1'b1 : d1 = d2
    );

    wire [EXP+FRA:0] e1; //exponential_1
    wire [EXP+FRA:0] e2; //exponential_2
    wire [EXP+FRA:0] sum; //exponential_1 + exponential_2
    wire [EXP+FRA:0] o1; //o1  = e1/sum
    wire [EXP+FRA:0] o2; //o2  = e2/sum
    wire [EXP+FRA:0] out;//out = o1 + o2
    wire [EXP+FRA:0] d1; //d1  = o1/out
    wire [EXP+FRA:0] d2; //d2  = o2/out

    wire s_axis_a_tready_1;
    wire exponential_1_valid; //exponential_1 valid
    wire exponential_1_ready; //exponential_1 ready out

    wire s_axis_a_tready_2;
    wire exponential_2_valid; //exponential_2 valid
    wire exponential_2_ready; //exponential_2 ready out

    wire sum_valid; //sum is valid
    wire sum_ready; //sum is ready to out

    wire o1_valid;
    wire o1_ready;
    wire o2_valid;
    wire o2_ready;

    wire out_valid;
    wire out_ready;

    wire d1_valid;
    wire d1_ready;
    wire d2_valid;
    wire d2_ready;

    //v1 turn to float
    exp_taylor #(
                   .EXP 		( EXP 		),
                   .FRA 		( FRA 		))
               exponential_1 (
                   .aclk                 		( aclk                 	),
                   .aresetn              		( aresetn              	),

                   .s_axis_tdata         		( s_axis_a_tdata        ),
                   .s_axis_tvalid        		( s_axis_a_tvalid       ),
                   .s_axis_tready        		( s_axis_a_tready       ),

                   .m_axis_result_tdata  		( e1  	                ),
                   .m_axis_result_tvalid 		( exponential_1_valid 	),
                   .flag                 		(                  		)
               );

    //v2 turn to float
    exp_taylor #(
                   .EXP 		( EXP 		),
                   .FRA 		( FRA 		))
               exponential_2 (
                   .aclk                 		( aclk                 	),
                   .aresetn              		( aresetn              	),

                   .s_axis_tdata         		( s_axis_b_tdata        ),
                   .s_axis_tvalid        		( s_axis_b_tvalid       ),
                   .s_axis_tready        		( s_axis_b_tready       ),

                   .m_axis_result_tdata  		( e2  	                ),
                   .m_axis_result_tvalid 		( exponential_2_valid 	),
                   .flag                 		(                  		)
               );

    //sum = e1 + e2
    tadd #(
             .EXP 		( EXP 		),
             .FRA 		( FRA 		))
         sum_e1_e2 (
             .aclk                 		( aclk                      ),
             .aresetn              		( aresetn              		),

             .s_axis_a_tdata       		( e1       		            ),
             .s_axis_a_tvalid      		( exponential_1_valid       ),
             .s_axis_a_tready      		( exponential_1_ready       ),

             .s_axis_b_tdata       		( e2       		            ),
             .s_axis_b_tvalid      		( exponential_2_valid       ),
             .s_axis_b_tready      		( exponential_2_ready       ),

             .m_axis_result_tdata  		( sum  		                ),
             .m_axis_result_tvalid 		( sum_valid 		        ),
             //.m_axis_result_tready 		( m_axis_result_tready 		),
             .flag                 		(                  		    )
         );

    reg [EXP+FRA:0] e1_div[3:0];
    reg             e1_div_valid[3:0];
    wire            e1_div_ready;

    reg [EXP+FRA:0] e2_div[3:0];
    reg             e2_div_valid[3:0];
    wire            e2_div_ready;

    integer i;

    always @(posedge aclk or posedge aresetn) begin
        if(aresetn) begin
            for(i=0;i<4;i=i+1) begin
                e1_div[i]       <= 1'b0;
                e2_div[i]       <= 1'b0;
                e1_div_valid[i] <= 1'b0;
                e2_div_valid[i] <= 1'b0;
            end
        end
        else begin
            e1_div[0]           <= e1;
            e2_div[0]           <= e2;
            e1_div_valid[0]     <= exponential_1_valid;
            e2_div_valid[0]     <= exponential_2_valid;
            for(i=0;i<3;i=i+1) begin
                e1_div[i+1]       <= e1_div[i];
                e2_div[i+1]       <= e2_div[i];
                e1_div_valid[i+1] <= e1_div_valid[i];
                e2_div_valid[i+1] <= e2_div_valid[i];
            end
        end
    end

    //o1 = e1/sum
    tdiv #(
             .EXP 		( EXP 		),
             .FRA 		( FRA 		))
         divide_operation_o1 (
             .aclk                 		( aclk                 	    ),
             .aresetn              		( aresetn              		),

             //S_AXIS_A
             .s_axis_a_tdata       		( e1_div[3]       		    ),
             .s_axis_a_tvalid      		( e1_div_valid[3]      		),
             .s_axis_a_tready      		( e1_div_ready      		),

             //S_AXIS_B
             .s_axis_b_tdata       		( sum       		        ),
             .s_axis_b_tvalid      		( sum_valid      		    ),
             .s_axis_b_tready      		( sum_ready      		    ),

             //S_AXIS_RESULT
             .m_axis_result_tdata  		( o1  		                ),
             .m_axis_result_tvalid 		( o1_valid           		),
             .flag                 		(                  		    )
         );

    //o2 = e2/sum
    tdiv #(
             .EXP 		( EXP 		),
             .FRA 		( FRA 		))
         divide_operation_o2 (
             .aclk                 		( aclk                 	    ),
             .aresetn              		( aresetn              		),

             //S_AXIS_A
             .s_axis_a_tdata       		( e2_div[3]       		    ),
             .s_axis_a_tvalid      		( e2_div_valid[3]      		),
             .s_axis_a_tready      		( e2_div_ready      		),

             //S_AXIS_B
             .s_axis_b_tdata       		( sum       		        ),
             .s_axis_b_tvalid      		( sum_valid      		    ),
             .s_axis_b_tready      		( sum_ready      		    ),

             //S_AXIS_RESULT
             .m_axis_result_tdata  		( o2  		                ),
             .m_axis_result_tvalid 		( o2_valid           		),
             .flag                 		(                  		    )
         );

    //sum = o1 + o2
    tadd #(
             .EXP 		( EXP 		),
             .FRA 		( FRA 		))
         sum_o1_o2 (
             .aclk                 		( aclk                      ),
             .aresetn              		( aresetn              		),

             .s_axis_a_tdata       		( o1       		            ),
             .s_axis_a_tvalid      		( o1_valid                  ),
             .s_axis_a_tready      		( o1_ready                  ),

             .s_axis_b_tdata       		( o2       		            ),
             .s_axis_b_tvalid      		( o2_valid                  ),
             .s_axis_b_tready      		( o2_ready                  ),

             .m_axis_result_tdata  		( out  		                ),
             .m_axis_result_tvalid 		( out_valid 		        ),
             //.m_axis_result_tready 		( m_axis_result_tready 		),
             .flag                 		(                  		    )
         );

    reg [EXP+FRA:0] o1_div[3:0];
    reg             o1_div_valid[3:0];
    wire            o1_div_ready;

    reg [EXP+FRA:0] o2_div[3:0];
    reg             o2_div_valid[3:0];
    wire            o2_div_ready;

    integer j;

    always @(posedge aclk or posedge aresetn) begin
        if(aresetn) begin
            for(j=0;j<4;j=j+1) begin
                o1_div[i]       <= 1'b0;
                o2_div[i]       <= 1'b0;
                o1_div_valid[i] <= 1'b0;
                o2_div_valid[i] <= 1'b0;
            end
        end
        else begin
            o1_div[0]           <= o1;
            o2_div[0]           <= o2;
            o1_div_valid[0]     <= o1_valid;
            o2_div_valid[0]     <= o2_valid;
            for(i=0;i<3;i=i+1) begin
                o1_div[i+1]       <= o1_div[i];
                o2_div[i+1]       <= o2_div[i];
                o1_div_valid[i+1] <= o1_div_valid[i];
                o2_div_valid[i+1] <= o2_div_valid[i];
            end
        end
    end

    //d1 = o1/out
    tdiv #(
             .EXP 		( EXP 		),
             .FRA 		( FRA 		))
         divide_operation_d1 (
             .aclk                 		( aclk                 	    ),
             .aresetn              		( aresetn              		),

             //S_AXIS_A
             .s_axis_a_tdata       		( o1_div[3]       		    ),
             .s_axis_a_tvalid      		( o1_div_valid[3]      		),
             .s_axis_a_tready      		( o1_div_ready      		),

             //S_AXIS_B
             .s_axis_b_tdata       		( out       		        ),
             .s_axis_b_tvalid      		( out_valid      		    ),
             .s_axis_b_tready      		( out_ready      		    ),

             //S_AXIS_RESULT
             .m_axis_result_tdata  		( d1  		                ),
             .m_axis_result_tvalid 		( d1_valid           		),
             .flag                 		(                  		    )
         );

    //d2 = o2/out
    tdiv #(
             .EXP 		( EXP 		),
             .FRA 		( FRA 		))
         divide_operation_d2 (
             .aclk                 		( aclk                 	    ),
             .aresetn              		( aresetn              		),

             //S_AXIS_A
             .s_axis_a_tdata       		( o2_div[3]       		    ),
             .s_axis_a_tvalid      		( o2_div_valid[3]      		),
             .s_axis_a_tready      		( o2_div_ready      		),

             //S_AXIS_B
             .s_axis_b_tdata       		( out       		        ),
             .s_axis_b_tvalid      		( out_valid      		    ),
             .s_axis_b_tready      		( out_ready      		    ),

             //S_AXIS_RESULT
             .m_axis_result_tdata  		( d2  		                ),
             .m_axis_result_tvalid 		( d2_valid           		),
             .flag                 		(                  		    )
         );

    //compare size
    wire sign_d1 = d1[EXP+FRA];
    wire sign_d2 = d2[EXP+FRA];

    wire [4:0] exponent_d1 = d1[EXP+FRA - 1 : EXP+FRA - 5];
    wire [4:0] exponent_d2 = d2[EXP+FRA - 1 : EXP+FRA - 5];

    wire [9:0] fraction_d1 = d1[EXP+FRA - 6 : EXP+FRA - 15];
    wire [9:0] fraction_d2 = d2[EXP+FRA - 6 : EXP+FRA - 15];

    always @(posedge aclk or posedge aresetn) begin
        if(aresetn) begin
            index  <= 1'b0;
            equal  <= 1'b0;
        end
        else if(d1_valid && d2_valid)begin
            if(sign_d1 ^ sign_d2) begin
                equal         <= 1'b0;
                case(sign_d1)
                    1'b0:
                        index <= 1'b1;
                    1'b1:
                        index <= 1'b0;
                    default:
                        ;
                endcase
            end
            else begin
                if(exponent_d1 > exponent_d2) begin
                    index  <= 1'b1;
                    equal  <= 1'b0;
                end
                else if(exponent_d1 < exponent_d2) begin
                    index  <= 1'b0;
                    equal  <= 1'b0;
                end
                else begin
                    if(fraction_d1 > fraction_d2) begin
                        index <= 1'b1;
                        equal <= 1'b0;
                    end
                    else if(fraction_d1 < fraction_d2) begin
                        index <= 1'b0;
                        equal <= 1'b0;
                    end
                    else begin
                        index  <= 1'bz;
                        equal <= 1'b1;
                    end
                end
            end
        end
    end

    reg final_valid;

    always @(posedge aclk) begin
        final_valid <= d1_valid && d2_valid;
    end

    assign m_axis_result_tvalid = final_valid;

endmodule
