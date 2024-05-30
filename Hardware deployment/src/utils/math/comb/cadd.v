module cadd #(
        parameter EXP = 5,
        parameter FRA = 10
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

    wire 			over;
    wire [EXP-1:0]	mexpo;
    wire [FRA:0]	ofracA;
    wire [FRA:0]	ofracB;

    pairStep #(
        .EXP 		( EXP 		),
        .FRA 		( FRA 		))
    u_pairStep(
        //ports
        .expoA  		( expoA  		), 
        .expoB  		( expoB  		),
        .ifracA 		( fracA 		), 
        .ifracB 		( fracB 		),

        .over   		( over   		),
        .oexpo  		( mexpo  		),
        .ofracA 		( ofracA 		),
        .ofracB 		( ofracB 		)
    );

    reg           sign;
    reg [FRA+1:0] ifrac;
    reg [EXP-1:0] iexpo;
    reg           subnormal;

    always @(*) begin
        if(aresetn)begin
            sign      <= 1'b0;
            iexpo     <= 1'b0;
            ifrac     <= 1'b0;
            subnormal <= 1'b0;
        end
        else if(expoA == 2**EXP-1 && expoB == 2**EXP-1)begin
            sign      <= 1'b0;
            iexpo     <= 2**EXP - 1'b1;
            ifrac     <= 1'b0;
            subnormal <= 1'b1;
        end
        else if(expoA == 2**EXP-1)begin
            sign      <= signA;
            iexpo     <= 2**EXP - 1'b1;
            ifrac     <= 1'b0;
            subnormal <= 1'b1;
        end
        else if(expoB == 2**EXP-1)begin
            sign      <= signB;
            iexpo     <= 2**EXP - 1'b1;
            ifrac     <= 1'b0;
            subnormal <= 1'b1;
        end
        else if(expoA == 1'b0 && expoB == 1'b0)begin
            subnormal <= 1'b1;
            if(signA ^ signB)begin
                if(ofracA > ofracB)begin
                    sign  <= signA;
                    ifrac <= ofracA - ofracB;
                end
                else begin
                    sign  <= signB;
                    ifrac <= ofracB - ofracA;
                end
            end
            else begin
                sign  <= signA | signB;
                ifrac <= ofracA + ofracB;
            end
            if(ifrac[FRA])
                iexpo <= mexpo + 1'b1;
            else 
                iexpo <= mexpo;
        end
        else begin
            subnormal <= 1'b0;
            iexpo     <= mexpo;
            if(signA ^ signB)begin
                if(ofracA > ofracB)begin
                    sign  <= signA;
                    ifrac <= ofracA - ofracB;
                end
                else begin
                    sign  <= signB;
                    ifrac <= ofracB - ofracA;
                end
            end
            else begin
                sign  <= signA | signB;
                ifrac <= ofracA + ofracB;
            end
        end
end

    wire [EXP-1:0]	r_oexpo;
    wire [FRA-1:0]	r_ofrac;
    wire [EXP-1:0]	d_oexpo;
    wire [FRA-1:0]	d_ofrac;
    wire [EXP-1:0]	oexpo;
    wire [FRA-1:0]	ofrac;

    wire [3:0]      cnt;

    normal #(
        .EXP 	(EXP),
        .FRA 	(FRA))
    u_normal(
        .iexpo		(iexpo),
        .ifrac		(ifrac),

        .oexpo		(r_oexpo),
        .ofrac		(r_ofrac),
        .cnt        (cnt    )
    );

assign d_oexpo = subnormal ? iexpo : ((iexpo < cnt) ? 0 : r_oexpo);
assign d_ofrac = subnormal ? ifrac : ((iexpo < cnt) ? ifrac[FRA:0] << (iexpo - 1) : r_ofrac);
assign oexpo   = d_oexpo;
assign ofrac   = (d_oexpo == 2**EXP-1) ? 0 : d_ofrac;

    cksp #(
        .EXP (EXP),
        .FRA (FRA))
    u_cksp(
        .expo	(oexpo),
        .frac	(ofrac),

        .flag 	(flag)
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

endmodule  //add_sub






