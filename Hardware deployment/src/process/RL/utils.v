
/***
@descriptenCn vector 
@state finish-test
***/
module vector #(
        parameter K = 4,
        parameter WIDTH = 16
    ) (
        input  clk,
        input  rst,

        input  avalid,
        input  [K*WIDTH-1:0]   A,

        input  bvalid,
        input  [K*WIDTH-1:0]   B,

        output yvalid,
        output [WIDTH-1:0]     Y
    );

    wire [K-1:0] mvalid;
    wire [WIDTH-1:0] mout [K-1:0] ;
    genvar n;

generate for(n = 0 ; n < K; n = n + 1) begin : Mult_array
            tmult u_mult(
                      .aclk(clk),
                      .aresetn(rst),

                      .s_axis_a_tvalid(avalid),
                      .s_axis_a_tdata(A[(n+1)*WIDTH-1 : n*WIDTH]),

                      .s_axis_b_tvalid(bvalid),
                      .s_axis_b_tdata(B[(n+1)*WIDTH-1 : n*WIDTH]),

                      .m_axis_result_tvalid(mvalid[n]),
                      .m_axis_result_tdata(mout[n])
                  );
        end
    endgenerate

    wire [K-2:0] valid;
    wire [WIDTH-1:0] aout [K-2:0] ;
    tadd u_add(
             .aclk(clk),
             .aresetn(rst),

             .s_axis_a_tvalid(mvalid[0]),
             .s_axis_a_tdata(mout[0]),

             .s_axis_b_tvalid(mvalid[1]),
             .s_axis_b_tdata(mout[1]),

             .m_axis_result_tvalid(valid[0]),
             .m_axis_result_tdata(aout[0])
         );

    integer i;

    reg [WIDTH - 1 : 0] r_mout[3:0];    //mout[2]
    reg                 r_mvalid[3:0];  //mvalid[2]

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for(i=0;i<4;i=i+1) begin
                r_mout[i]   <= 1'b0;
                r_mvalid[i] <= 1'b0;
            end
        end
        else begin
            r_mout[0]   <= mout[2];
            r_mvalid[0] <= mvalid[2];
            for(i=0;i<3;i=i+1) begin
                r_mout[i+1]   <= r_mout[i];
                r_mvalid[i+1] <= r_mvalid[i];
            end
        end
    end

    integer j;

    reg [WIDTH - 1 : 0] d_mout[7:0]; //mout[3]
    reg                 d_mvalid[7:0];  //mvalid[3]

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for(j=0;j<8;j=j+1) begin
                d_mout[j]   <= 1'b0;
                d_mvalid[j] <= 1'b0;
            end
        end
        else begin
            d_mout[0]   <= mout[3];
            d_mvalid[0] <= mvalid[3];
            for(j=0;j<7;j=j+1) begin
                d_mout[j+1]   <= d_mout[j];
                d_mvalid[j+1] <= d_mvalid[j];
            end
        end
    end

    tadd r_add(
             .aclk(clk),
             .aresetn(rst),

             .s_axis_a_tvalid(r_mvalid[3]),
             .s_axis_a_tdata(r_mout[3]),

             .s_axis_b_tvalid(valid[0]),
             .s_axis_b_tdata(aout[0]),

             .m_axis_result_tvalid(valid[1]),
             .m_axis_result_tdata(aout[1])
         );

    tadd d_add(
             .aclk(clk),
             .aresetn(rst),

             .s_axis_a_tvalid(d_mvalid[7]),
             .s_axis_a_tdata(d_mout[7]),

             .s_axis_b_tvalid(valid[1]),
             .s_axis_b_tdata(aout[1]),

             .m_axis_result_tvalid(valid[2]),
             .m_axis_result_tdata(aout[2])
         );

    assign yvalid = valid[K-2];
    assign Y = aout[K-2];

endmodule  //vector

module ser2par #(
        parameter WIDTH = 16
    ) (
        input   clk,
        input   rst,

        input   ivalid,
        input   [WIDTH-1:0]   in,

        output  ovalid,
        output  [4*WIDTH-1:0] out
    );

    // 灏嗗洓涓緭鍏ヨ繘琛屾嫾鎺�
    reg [4*WIDTH-1:0] rout;
    reg [1:0] cnt;
    reg valid;

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            cnt <= 0;
            rout <= 0;
            valid <= 0;
        end
        else if (ivalid) begin
            if (cnt == 3) begin
                valid <= 1'b1;
            end
            else begin
                valid <= 1'b0;
            end
            cnt <= cnt + 1;
            rout[WIDTH-1:0] <= in;
            rout[2*WIDTH-1:WIDTH] <= rout[WIDTH-1:0];
            rout[3*WIDTH-1:2*WIDTH] <= rout[2*WIDTH-1:WIDTH];
            rout[4*WIDTH-1:3*WIDTH] <= rout[3*WIDTH-1:2*WIDTH];
        end
        else begin
            cnt <= 0;
            rout <= 0;
            valid <= 0;
        end
    end

    assign ovalid = valid;
    assign out = rout;

endmodule //ser2par

