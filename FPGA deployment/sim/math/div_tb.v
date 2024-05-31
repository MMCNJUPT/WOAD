module div_tb();

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
    reg     [EXP+FRA:0] s_axis_a_tdata;
    reg     [EXP+FRA:0] s_axis_b_tdata;

    reg					s_axis_a_tvalid;
    reg					s_axis_b_tvalid;

    wire 				s_axis_a_tready;
    wire 				s_axis_b_tready;

    wire [EXP+FRA:0]	m_axis_tdata;
    wire 				m_axis_tvalid;
    reg                 m_axis_tready;

    initial begin
        s_axis_a_tdata  = 16'h3266;
        s_axis_b_tdata  = 16'h2e66;
        s_axis_a_tvalid = 1'b0;
        s_axis_b_tvalid = 1'b0;
        m_axis_tready   = 1'b0;
		#100
		s_axis_a_tvalid = 1'b1;
        s_axis_b_tvalid = 1'b1;
        m_axis_tready   = 1'b1;
    end

    tdiv #(
            .EXP 		( EXP 		),
            .FRA 		( FRA 		))
        u_div(
            //global
            .aclk                 		( sys_clk                 	),
            .aresetn              		( sys_rst              		),

            //S_AXIS_A
            .s_axis_a_tdata       		( s_axis_a_tdata       		),
            .s_axis_a_tvalid      		( s_axis_a_tvalid      		),
            .s_axis_a_tready      		( s_axis_a_tready      		),

            //S_AXIS_B
            .s_axis_b_tdata       		( s_axis_b_tdata       		),
            .s_axis_b_tvalid      		( s_axis_b_tvalid      		),
            .s_axis_b_tready      		( s_axis_b_tready      		),

            //S_AXIS_RESULT
            .m_axis_tdata  		        ( m_axis_tdata  		    ),
            .m_axis_tvalid 		        ( m_axis_tvalid 		    ),
            .m_axis_tready              ( m_axis_tready             )
        );

endmodule  //TOP
