module tadd #(
        parameter EXP = 5,
        parameter FRA = 10
    ) (
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
        //m_axis_result_tdata: Output specifications, non-specified number output to 0
        output [EXP+FRA:0] m_axis_result_tdata,
        //the output is invalid
        output             m_axis_result_tvalid,
        //input              m_axis_result_tready,

        //Zero: flag[0]=1  Inf: flag[1] = 1  NaN: flag[2] = 1
        output [2:0]       flag
    );

    wire 	        signA, signB;
    wire [EXP-1:0]	expoA, expoB;
    wire [FRA:0]	fracA, fracB;
    wire            M_A_tvalid,M_B_tvalid;

    //disassemble for each part of the floating point number
    //One clock cycles are consumed
    unpack_sequential #(
                          .EXP (EXP),
                          .FRA (FRA))
                      u_unpack_sequential(
                          .aclk           (aclk),
                          .aresetn        (aresetn),

                          .inA            (s_axis_a_tdata),
                          .S_A_tvalid     (s_axis_a_tvalid),

                          .inB            (s_axis_b_tdata),
                          .S_B_tvalid     (s_axis_b_tvalid),

                          .signA          (signA),
                          .expoA          (expoA),
                          .fracA          (fracA),
                          .M_A_tvalid     (M_A_tvalid),

                          .signB          (signB),
                          .expoB          (expoB),
                          .fracB          (fracB),
                          .M_B_tvalid     (M_B_tvalid)
                      );

    wire 	        over;
    wire [EXP-1:0]	mexpo;
    wire [FRA:0]	ofracA;
    wire [FRA:0]	ofracB;
    wire            M_pairStep_tvalid;
    wire            M_pairStep_A_tvalid;
    wire            M_pairStep_B_tvalid;

    //pairStep
    //One clock cycles are consumed
    pairStep_sequential #(
                            .EXP (EXP),
                            .FRA (FRA))
                        u_pairStep_sequential(
                            .aclk               (aclk),
                            .aresetn            (aresetn),

                            .expoA              (expoA),
                            .S_A_tvalid         (M_A_tvalid),

                            .expoB              (expoB),
                            .S_B_tvalid         (M_B_tvalid),

                            .ifracA             (fracA),
                            .ifracB             (fracB),

                            .over               (over),
                            .oexpo              (mexpo),
                            .M_tvalid           (M_pairStep_tvalid),

                            .ofracA             (ofracA),
                            .M_A_tvalid         (M_pairStep_A_tvalid),

                            .ofracB             (ofracB),
                            .M_B_tvalid         (M_pairStep_B_tvalid)
                        );

    reg           r_signA;
    reg           r_signB;
    reg [EXP-1:0] r_expoA, r_expoB;
    reg [FRA:0]	  r_fracA, r_fracB;

    always @(posedge aclk or posedge aresetn) begin
        if(aresetn) begin
            r_signA <= 1'b0;
            r_signB <= 1'b0;
            r_expoA <= 1'b0;
            r_expoB <= 1'b0;
            r_fracA <= 1'b0;
            r_fracB <= 1'b0;
        end
        else begin
            r_signA <= signA;
            r_signB <= signB;
            r_expoA <= expoA;
            r_expoB <= expoB;
            r_fracA <= fracA;
            r_fracB <= fracB;
        end
    end

    reg [2:0] state; 
    reg [2:0] r_state,d_state;
    // 001: subnormal + subnormal  010: normal  100: lnf 

    //second
    always @(*) begin
        if(aresetn)
            state <= 3'b000;
        else if(r_expoA == 1'b0 && r_expoB == 1'b0)
            state <= 3'b001;
        else if(r_expoA == 2**EXP-1 || r_expoB == 2**EXP-1)
            state <= 3'b100;
        else begin
            state <= 3'b010;
        end
    end

    always @(posedge aclk) begin
        r_state <= state;
        d_state <= r_state;
    end

    //third
    reg           r_sign;
    reg [EXP-1:0] iexpo;
    reg [FRA+1:0] ifrac;
    reg           r_valid;

    always @(posedge aclk or posedge aresetn) begin
        if(aresetn)begin
            r_sign  <= 1'b0;
            iexpo   <= 1'b0;
            ifrac   <= 1'b0;
            r_valid <= 1'b0;
        end
        else if(r_expoA == 2**EXP-1 && r_expoB == 2**EXP-1)begin
            r_sign  <= 1'b0;
            iexpo   <= 2**EXP - 1'b1;
            ifrac   <= 1'b0;
            r_valid <= M_pairStep_A_tvalid && M_pairStep_B_tvalid && M_pairStep_tvalid;
        end
        else if(r_expoA == 2**EXP-1)begin
            r_sign  <= r_signA;
            iexpo   <= 2**EXP - 1'b1;
            ifrac   <= 1'b0;
            r_valid <= M_pairStep_A_tvalid && M_pairStep_B_tvalid && M_pairStep_tvalid;
        end
        else if(r_expoB == 2**EXP-1)begin
            r_sign  <= r_signB;
            iexpo   <= 2**EXP - 1'b1;
            ifrac   <= 1'b0;
            r_valid <= M_pairStep_A_tvalid && M_pairStep_B_tvalid && M_pairStep_tvalid;
        end
        else begin
            iexpo   <= mexpo;
            r_valid <= M_pairStep_A_tvalid && M_pairStep_B_tvalid && M_pairStep_tvalid;
            if(r_signA ^ r_signB)begin
                if(ofracA > ofracB)begin
                    r_sign <= r_signA;
                    ifrac  <= ofracA - ofracB;
                end
                else begin
                    r_sign <= r_signB;
                    ifrac  <= ofracB - ofracA;
                end
            end
            else begin
                r_sign <= r_signA | r_signB;
                ifrac  <= ofracA + ofracB;
            end
        end
    end

    reg           sign;
    reg [EXP-1:0] r_iexpo;
    reg [FRA+1:0] r_ifrac;

    always @(posedge aclk) begin
        sign    <= r_sign;
        r_iexpo <= iexpo;
        r_ifrac <= ifrac;
    end

    wire [EXP-1:0]	r_oexpo;
    wire [FRA-1:0]	r_ofrac;
    reg  [EXP-1:0]	d_oexpo;
    reg  [FRA-1:0]	d_ofrac;
    wire [EXP-1:0]	oexpo;
    wire [FRA-1:0]	ofrac;
    wire            d_valid;
    //fourth
    wire [3:0] cnt;

    normal_sequential #(
                          .EXP (EXP),
                          .FRA (FRA))
    u_normal_sequential(
                          .aclk           (aclk),
                          .aresetn        (aresetn),

                          .iexpo          (iexpo),
                          .ifrac          (ifrac),
                          .S_tvalid       (r_valid),

                          .oexpo          (r_oexpo),
                          .ofrac          (r_ofrac),
                          .M_tvalid       (d_valid),
                          .cnt            ( cnt   )
                      );

    always @(*) begin
        case(d_state)
            3'b000:begin
                d_oexpo <= 1'b0;
                d_ofrac <= 1'b0;
            end
            3'b001:begin
                if(r_ifrac[FRA])begin
                    d_oexpo <= r_iexpo + 1'b1;
                    d_ofrac <= r_ifrac[FRA:0];
                end
                else begin
                    d_oexpo <= r_iexpo;
                    d_ofrac <= r_ifrac[FRA:0];
                end
            end
            3'b010:begin
                if(r_iexpo < cnt)begin
                    d_oexpo <= 1'b0;
                    d_ofrac <= r_ifrac[FRA:0] << (r_iexpo - 1);
                end
                else begin
                    d_oexpo <= r_oexpo;
                    d_ofrac <= r_ofrac;
                end
            end
            3'b100:begin
                d_oexpo <= 2**EXP - 1'b1;
                d_ofrac <= 1'b0;
            end
        endcase
    end

assign oexpo   = d_oexpo;
assign ofrac   = (d_oexpo == 2**EXP-1) ? 0 : d_ofrac; 

//Determine whether there are special numbers
    cksp #(
             .EXP (EXP),
             .FRA (FRA))
         u_cksp(
             .expo	   (oexpo),
             .frac	   (ofrac),

             .flag 	   (flag)
         );

    //Combine each part of the floating point number
    pack #(
             .EXP 		( EXP 		),
             .FRA 		( FRA 		))
         u_pack(
             //ports
             .out  		( m_axis_result_tdata    ),
             .sign 		( sign 		             ),
             .expo 		( oexpo 	             ),
             .frac 		( ofrac 	             )
         );

    assign s_axis_a_tready      = s_axis_a_tvalid;
    assign s_axis_b_tready      = s_axis_b_tvalid;
    assign m_axis_result_tvalid = d_valid;

endmodule
