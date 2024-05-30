module tdiv #(
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
        //input            m_axis_result_tready,

        //Zero: flag[0]=1  Inf: flag[1] = 1  NaN: flag[2] = 1
        output [2:0]       flag
    );

    wire 	        signA, signB;
    wire [EXP-1:0]	expoA, expoB;
    wire [FRA:0]	fracA, fracB;
    wire            M_unpack_A_tvalid;
    wire            M_unpack_B_tvalid;

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
        .M_A_tvalid     (M_unpack_A_tvalid),

        .signB          (signB),
        .expoB          (expoB),
        .fracB          (fracB),
        .M_B_tvalid     (M_unpack_B_tvalid)
    );//one

    reg              r_sign;
    reg              d_sign;
    reg              sign;

    reg  [EXP:0]     r_iexpo;
    reg  [EXP-1:0]   iexpo;

    wire [2*FRA+1:0] fraction;
    reg  [FRA+1:0]   r_ifrac;
    reg  [FRA:0]     ifrac;

    reg              r_S_normal_valid;
    reg              S_normal_valid;

    assign fraction = {fracA,{(FRA+1){1'b0}}};

    always @(posedge aclk or posedge aresetn) begin
        if(aresetn)begin
            r_sign           <= 1'b0;
            r_iexpo          <= 1'b0;
            r_ifrac          <= 1'b0;
            r_S_normal_valid <= 1'b0;
        end
        else begin
            r_S_normal_valid <= M_unpack_A_tvalid && M_unpack_B_tvalid;
            //whether A is infinitely small
            if(expoA == {(EXP){1'b0}})begin
                //A:small B:small output:big
                if(expoB == {(EXP){1'b0}})begin
                    r_sign  <= signA ^ signB;
                    r_iexpo <= 2**EXP - 1'b1;
                    r_ifrac <= {1'b1,{(FRA+1){1'b0}}};
                end
                //A:small B:other output:small
                else begin
                    r_sign  <= 1'b0;
                    r_iexpo <= 1'b0;
                    r_ifrac <= 1'b0;
                end
            end
            //A:big B:big output:big
             else if(expoA == 2**EXP - 1)begin
                 r_sign     <= signA ^ signB;
                 r_iexpo    <= 2**EXP - 1'b1;
                 r_ifrac    <= {1'b1,{(FRA+1){1'b0}}};
             end
             //B:small output: big
             else if(expoB == {(EXP){1'b0}})begin
                 r_sign     <= signA ^ signB;
                 r_iexpo    <= 2**EXP - 1'b1;
                 r_ifrac <= {1'b1,{(FRA+1){1'b0}}};
             end
             //B:big output: small
             else if(expoB == 2**EXP - 1)begin
                 r_sign     <= 1'b0;
                 r_iexpo    <= 1'b0;
                 r_ifrac    <= 1'b0;
             end
             else begin
                 r_sign     <= signA ^ signB;
                 r_iexpo    <= expoA - expoB;
                 r_ifrac    <= fraction/fracB;
             end
        end
    end

    always @(posedge aclk or posedge aresetn) begin
        if(aresetn)begin
            d_sign         <= 1'b0;
            iexpo          <= 1'b0;
            ifrac          <= 1'b0;
            S_normal_valid <= 1'b0;
        end
        else begin
            S_normal_valid <= r_S_normal_valid;
            if(r_iexpo + 2**(EXP-1) <= 1)begin
                d_sign     <= 1'b0;
                iexpo      <= 1'b0;
                ifrac      <= 1'b0;
            end
            else begin
                d_sign     <= r_sign;
                iexpo      <= r_iexpo + (2**(EXP-1)-1);
                if(r_ifrac[0])begin
                    ifrac  <= r_ifrac[FRA + 1 : 1] + 1'b1;
                end
                else begin
                    ifrac  <= r_ifrac[FRA + 1 : 1];
                end
            end
        end
    end

    always @(posedge aclk) begin
        sign    <= d_sign;
    end

    wire [FRA - 1 : 0] ofrac;
    wire [EXP - 1 : 0] oexpo;
    wire               M_normal_valid;

    normal_tmult #(
        .EXP (EXP),
        .FRA (FRA))
    u_normal_tmult(
        .aclk           (aclk),
        .aresetn        (aresetn),

        .iexpo          (iexpo),
        .ifrac          (ifrac),
        .S_tvalid       (S_normal_valid),

        .oexpo          (oexpo),
        .ofrac          (ofrac),
        .M_tvalid       (M_normal_valid)
    );

    cksp #(
        .EXP (EXP),
        .FRA (FRA))
    u_cksp(
        .expo	  (oexpo),
        .frac	  (ofrac),

        .flag 	  (flag)
    );

    pack #(
        .EXP 		( EXP 		),
        .FRA 		( FRA 		))
    u_pack(
        //ports
        .out  		( m_axis_result_tdata),
        .sign 		( sign 		         ),
        .expo 		( oexpo 	         ),
        .frac 		( ofrac 	         )
    );

    assign s_axis_a_tready = s_axis_a_tvalid;
    assign s_axis_b_tready = s_axis_b_tvalid;
    assign m_axis_result_tvalid = M_normal_valid;

endmodule