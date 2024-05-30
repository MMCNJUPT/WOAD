module cmult #(
        parameter EXP = 4,
        parameter FRA = 3
    ) (
        input			   aresetn,
        input			   valid,
        input  [EXP+FRA:0] A,
        input  [EXP+FRA:0] B,
        output [EXP+FRA:0] Y,
        output [2:0]       flag
    );

    wire 	        signA, signB;
    wire [EXP-1:0]	expoA, expoB;
    wire [FRA:0]	fracA, fracB;

    unpack #(
        .EXP 		( EXP ),
        .FRA 		( FRA ))
    u_unpackA(
        //ports
        .inA   ( A     ), .inB   ( B     ),
        .signA ( signA ), .expoA ( expoA ), .fracA ( fracA ),
        .signB ( signB ), .expoB ( expoB ), .fracB ( fracB )
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

    always @(*) begin
        if(aresetn)begin
            r_sign   <= 1'b0;
            r_iexpo  <= 1'b0;
            fraction <= 1'b0;
        end
        else begin
            r_sign      <= signA ^ signB;
            r_iexpo     <= expoA + expoB;
            fraction    <= fracA * fracB;
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
        else if(r_iexpo >= 2**(EXP) + 2**(EXP-1) - 1 || expoA == 2**EXP-1 || expoB == 2**EXP-1)begin
            state <= 3'b010;
        end
        else begin
            state <= 3'b100;
        end
    end

    always @(*) begin
            case(state)
                3'b000:begin
                    d_sign             <= 1'b0;
                    iexpo              <= 1'b0;
                    ifrac              <= 1'b0;
                    normal             <= 1'b0;
                end
                3'b001:begin
                    d_sign             <= r_sign;
                    iexpo              <= 1'b0;
                    normal             <= 1'b0;
                    if(expoA == 1'b0 || expoB == 1'b0)begin
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
                    normal             <= 1'b1;
                end
                3'b100:begin
                    d_sign             <= r_sign;
                    if(expoA == 1'b0 || expoB == 1'b0)
                        iexpo          <= r_iexpo - (2**(EXP-1) - 3);
                    else 
                        iexpo          <= r_iexpo - (2**(EXP-1) - 2);
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

    normal_mult #(
        .EXP 	(EXP),
        .FRA 	(FRA))
    u_normal(
        .iexpo		(iexpo),
        .ifrac		(ifrac[FRA:0]),

        .oexpo		(r_oexpo),
        .ofrac		(r_ofrac)
    );

    reg [FRA     : 0] d_ofrac;
    reg [EXP - 1 : 0] d_oexpo;
    reg               r_normal;

    always @(*) begin
        sign        <= d_sign;
        d_oexpo     <= iexpo;
        if(ifrac[FRA])begin
            d_ofrac <= ifrac[2*FRA+1:FRA+1] + 1'b1;
        end
        else begin
            d_ofrac <= ifrac[2*FRA+1:FRA+1];
        end 
        r_normal    <= normal;
    end

    wire [FRA - 1 : 0] ofrac;
    wire [EXP - 1 : 0] oexpo;

    assign oexpo = r_normal ? r_oexpo : d_oexpo;
    assign ofrac = r_normal ? r_ofrac : d_ofrac[FRA:FRA-9];

    cksp #(
        .EXP (EXP),
        .FRA (FRA))
    u_cksp(
        .expo	(oexpo),
        .frac	(ofrac),

        .flag	(flag)
    );

    pack #(
        .EXP 		( EXP 		),
        .FRA 		( FRA 		))
    u_pack(
        //ports
        .out  		( Y  		),
        .sign 		( sign 		),
        .expo 		( oexpo 	),
        .frac 		( ofrac 	)
    );

endmodule  //mult
