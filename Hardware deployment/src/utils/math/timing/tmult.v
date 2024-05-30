module tmult #(
        parameter EXP = 5,
        parameter FRA = 10
    ) (
        input              aclk,
        input              aresetn,

        //S_AXIS_A
        input  [EXP+FRA:0] s_axis_a_tdata,
        input              s_axis_a_tvalid,
        output             s_axis_a_tready,

        //S_AXIS_B
        input  [EXP+FRA:0] s_axis_b_tdata,
        input              s_axis_b_tvalid,
        output             s_axis_b_tready,

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
 
    reg             r_sign;
    reg             d_sign;
    reg             sign;

    reg [EXP:0]     r_iexpo;
    reg [EXP-1:0]   iexpo;

    reg [2*FRA+1:0] fraction;
    reg [2*FRA+1:0] ifrac;

    reg             r_valid;
    reg             d_valid;

    reg [EXP-1:0]	r_expoA, r_expoB;

    always @(posedge aclk or posedge aresetn) begin
        if(aresetn)begin
            r_sign      <= 1'b0;
            r_iexpo     <= 1'b0;
            fraction    <= 1'b0;
            r_valid     <= 1'b0;
            r_expoA     <= 1'b0;
            r_expoB     <= 1'b0;
        end
        else begin
            r_sign      <= signA ^ signB;
            r_iexpo     <= expoA + expoB;
            fraction    <= fracA * fracB;
            r_valid     <= M_A_tvalid && M_B_tvalid;
            r_expoA     <= expoA;
            r_expoB     <= expoB;
        end
    end

    reg normal; //normal = 1'b1: normal number/inf   normal = 1'b0: subnormal number

    reg [2:0] state; // 001: subnormal number  010: inf  100: normal number

    always @(*) begin
        if(aresetn)begin
            state <= 3'b000;
        end
        else if(r_iexpo < 2**(EXP-1) - 1)begin
            state <= 3'b001;
        end
        else if(r_iexpo == 2**(EXP-1) - 1)begin
            if(fraction[2*FRA+1])
                state <= 3'b100;
            else 
                state <= 3'b001;
        end
        else if(r_iexpo >= 2**(EXP) + 2**(EXP-1) - 1 || r_expoA == 2**EXP-1 || r_expoB == 2**EXP-1)begin
            state <= 3'b010;
        end
        else begin
            state <= 3'b100;
        end
    end

    always @(posedge aclk) begin
            case(state)
                3'b000:begin
                    d_sign             <= 1'b0;
                    iexpo              <= 1'b0;
                    ifrac              <= 1'b0;
                    d_valid            <= 1'b0;
                    normal             <= 1'b0;
                end
                3'b001:begin
                    d_sign             <= r_sign;
                    iexpo              <= 1'b0;
                    d_valid            <= r_valid;
                    normal             <= 1'b0;
                    if(r_expoA == 1'b0 || r_expoB == 1'b0)begin
                        ifrac          <= fraction >> (2**(EXP-1) - 3 - r_iexpo);
                    end
                    else begin
                        if(r_iexpo <= 2**(EXP-1) - 2)begin
                            ifrac      <= fraction >> (2**(EXP-1) - 2 - r_iexpo);
                        end
                    else begin
                            ifrac      <= fraction << (r_iexpo - 2**(EXP-1) + 2);
                        end
                    end
                end
                3'b010:begin
                    d_sign             <= r_sign;
                    iexpo              <= 2**EXP - 1'b1;
                    ifrac[FRA:0]       <= {1'b1,{(FRA){1'b0}}};
                    d_valid            <= r_valid;
                    normal             <= 1'b1;
                end
                3'b100:begin
                    d_sign             <= r_sign;
                    if(r_expoA == 1'b0 || r_expoB == 1'b0)
                        iexpo          <= r_iexpo - (2**(EXP-1) - 3);
                    else 
                        iexpo          <= r_iexpo - (2**(EXP-1) - 2);
                    d_valid            <= r_valid;
                    normal             <= 1'b1;
                    if(fraction[FRA])begin
                        ifrac[FRA:0]   <= fraction[2*FRA+1:FRA+1] + 1'b1;
                    end
                    else begin
                        ifrac[FRA:0]   <= fraction[2*FRA+1:FRA+1];
                    end 
                end
            endcase
        end

    wire [FRA - 1 : 0] r_ofrac;
    wire [EXP - 1 : 0] r_oexpo;
    wire               normal_valid;

    normal_tmult #(
        .EXP (EXP),
        .FRA (FRA))
    u_normal_tmult(
        .aclk           (aclk),
        .aresetn        (aresetn),

        .iexpo          (iexpo),
        .ifrac          (ifrac[FRA:0]),
        .S_tvalid       (d_valid),

        .oexpo          (r_oexpo),
        .ofrac          (r_ofrac),
        .M_tvalid       (normal_valid)
    );

    reg [FRA     : 0] d_ofrac;
    reg [EXP - 1 : 0] d_oexpo;
    reg               r_normal;
    reg               subnormal_valid;

    always @(posedge aclk) begin
        sign        <= d_sign;
        d_oexpo     <= iexpo;
        if(ifrac[FRA])begin
            d_ofrac <= ifrac[2*FRA+1:FRA+1] + 1'b1;
        end
        else begin
            d_ofrac <= ifrac[2*FRA+1:FRA+1];
        end 
        r_normal        <= normal;
        subnormal_valid <= d_valid;
    end

    wire [FRA - 1 : 0] ofrac;
    wire [EXP - 1 : 0] oexpo;

    assign oexpo = r_normal ? r_oexpo : d_oexpo;
    assign ofrac = r_normal ? r_ofrac : d_ofrac[FRA:FRA-9];
    assign m_axis_result_tvalid = r_normal ? normal_valid : subnormal_valid;

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

    assign s_axis_a_tready      = s_axis_a_tvalid;
    assign s_axis_b_tready      = s_axis_b_tvalid;

endmodule  //mult
