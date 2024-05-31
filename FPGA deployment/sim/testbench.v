module testbench();

	wire	[7:0]	Y;
	reg 	[7:0]	A,B;

    mult #(
        .EXP 		( 4 		),
        .FRA 		( 3 		))
    uut(
        //ports
        // .t      ( 0         ),
        .A 		( A 		),
        .B 		( B 		),
        .Y 		( Y 		)
    );

	initial
		begin
			A = 8'b0_0110_010; //0.5		
			B = 8'b1_0100_100; //0.4375
        //  Y = 8'b0_110_1000
			#50
			$finish;
		end


endmodule  //TOP
