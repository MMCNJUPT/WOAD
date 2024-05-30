module GEMMR3 #(
        parameter    K = 2,
        parameter    WIDTH = 16
    ) (
        input      clk,
        input      rst,

        input              gvalid,
        input              ivalid,
        input  [WIDTH-1:0] in,

        output             ovalid,
        output [WIDTH-1:0] out
    );

    // ------ 串并转换�?1?7?4位拼�?1?7? ------ //
    // finish-test
    wire 	svalid;
    wire [4*WIDTH-1:0]	pout;

    ser2par #(
                .WIDTH 		( WIDTH ))
            u_ser2par(
                //ports
                .clk    ( clk    ),
                .rst    ( rst    ),

                .ivalid ( ivalid ),
                .in     ( in     ),

                .ovalid ( svalid ),
                .out    ( pout   )
            );

    // ------ 由于之后�?1?7?要多次调用，在此进行存储 ------ //
    reg full;
    reg [5:0] cnt;
    reg [4*WIDTH-1:0] gemm[63:0];

    integer i;
    initial begin
        for (i = 0; i<64; i=i+1) begin
            gemm[i] = 0;
        end
    end

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            full <= 0;
            cnt <= 0;
        end
        else if(gvalid) begin
            if (svalid) begin
                if (cnt == 63) begin // �?1?7?要加解锁装置
                    full <= 1'b1;
                end
                else begin
                    cnt <= cnt + 1;
                end
            end
        end
        else begin
            full <= 0;
            cnt <= 0;
        end
    end

    always @(posedge clk) begin
        if(!full) begin
            gemm[cnt] <= pout;
        end
    end

    reg        addr;
    reg [63:0] data;
    reg [5:0]  index;
    reg [2:0]  finish;
    reg        r_finish[14:0];
    reg        d_finish[14:0];

    integer m;
    integer a;

    /*
    finish[0]   : all of the data and param are be inputed to the vector
    finish[1]   : the results of vector have been inputed to u0_add 
    d_finish[3] : matrix multiplication completed
    */

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            data <= 0;
            addr <= 0;
            index <= 0;
            finish <= 0;
            for(m=0;m<15;m=m+1) begin
                r_finish[m] <= 1'b0;
            end
            for(a=0;a<15;a=a+1) begin
                d_finish[a] <= 1'b0;
            end
        end
        else if(gvalid) begin
            if (full) begin
                if (addr == 1) begin
                    if (index != 63) begin
                        index <= index + 1;
                    end
                    else begin
                        finish[0] <= 1'b1;
                    end
                end
                addr <= addr + 1;
                data <= gemm[index];
            end
            r_finish[0] <= finish[0];
            for(m=0;m<14;m=m+1) begin
                r_finish[m+1] <= r_finish[m];
            end
            finish[1]   <= r_finish[14];
            d_finish[0] <= finish[1];
            for(a=0;a<14;a=a+1) begin
                d_finish[a+1] <= d_finish[a];
            end
            finish[2]   <= d_finish[14];
        end
        else begin
            data <= 0;
            addr <= 0;
            index <= 0;
            finish <= 0;
            for(m=0;m<15;m=m+1) begin
                r_finish[m] <= 1'b0;
            end
            for(a=0;a<15;a=a+1) begin
                d_finish[a] <= 1'b0;
            end
        end
    end

    wire [63:0] param;

    gemm3B u_gemm3B (
               .clk(clk),    // input wire clka
               .rst(rst),

               .addr({index,addr}),  // input wire [6 : 0] addra
               .data(param)  // output wire [63 : 0] douta
           );

    reg cvalid;
    reg r_addr[14:0];
    reg oaddr;

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            cvalid <= 0;
            oaddr <= 0;
        end
        else begin
            cvalid <= full;
            oaddr <= r_addr[14];
        end
    end

    integer k;

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for(k=0;k<15;k=k+1) begin
                r_addr[k] <= 1'b0;
            end
        end
        else begin
            r_addr[0]       <= addr;
            for(k=0;k<14;k=k+1) begin
                r_addr[k+1] <= r_addr[k];
            end
        end
    end

    wire 	yvalid;
    wire [WIDTH-1:0]	Y;

    vector #(
               .K     		( 4  		),
               .WIDTH 		( 16 		))
           u_vector(
               //ports
               .clk         ( clk     ),
               .rst         ( rst     ),
               .avalid 		( cvalid  ),
               .A      		( data    ),
               .bvalid 		( cvalid  ),
               .B      		( param   ),
               .yvalid 		( yvalid  ),
               .Y      		( Y       )
           );

    reg [WIDTH-1:0] result[1:0];
    integer n;
    initial begin
        for (n = 0; n<2; n=n+1) begin
            result[n] = 0;
        end
    end

    // wire valid;
    wire [WIDTH-1:0] ores;
    
    cadd #(
                .EXP 		( 5 		),
                .FRA 		( 10 		))
        u_add_sub(
                //ports
                .aresetn    ( rst                  ),
                .valid      ( yvalid               ),
                .A 		    ( Y 		           ),
                .B 		    ( result[oaddr - 1'b1] ),
                .Y 		    ( ores    		       ),
                .flag       (                      )
            );//the first

    /*
    tadd u0_add(
             .aclk(clk),
             .aresetn(rst),

             .s_axis_a_tvalid(yvalid),
             .s_axis_a_tdata(Y),

             .s_axis_b_tvalid(yvalid),
             .s_axis_b_tdata(result[oaddr - 1'b1]),

             .m_axis_result_tvalid(),
             .m_axis_result_tdata(ores)
         );
    */
    integer j;

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for(j=0;j<2;j=j+1) begin
                result[j] <= {(WIDTH){1'b0}};
            end
        end
        else if(gvalid) begin
            if(cvalid & (~finish[1])) begin
                result[oaddr - 1'b1] <= ores;
            end
        end
        else begin
            for(j=0;j<2;j=j+1) begin
                result[j] <= {(WIDTH){1'b0}};
            end
        end
    end

    wire [WIDTH-1:0] cparam;
    gemm3C u_gemm3C(
               //ports
               .addr 		( oaddr ),
               .data 		( cparam)
           );

    // wire valid;
    wire [WIDTH-1:0] cout;
    wire [WIDTH-1:0] cdata = result[oaddr];
    tadd u1_add(
             .aclk(clk),
             .aresetn(rst),

             .s_axis_a_tvalid(finish[1]),
             .s_axis_a_tdata(cparam),

             .s_axis_b_tvalid(finish[1]),
             .s_axis_b_tdata(cdata),

             .m_axis_result_tvalid(),
             .m_axis_result_tdata(cout)
         );

    reg [WIDTH-1:0] rout;
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            rout <= 0;
        end
        else if(gvalid) begin
            if (d_finish[3]) begin
                rout <= cout;
            end
        end
        else begin
            rout <= 0;
        end
    end
    assign out = rout;
    assign ovalid = d_finish[4];

endmodule
