module GEMMR1 #(
        parameter    LENGTH = 256,
        parameter    WIDTH = 16,
        parameter    DEPTH = 2
    ) (
        input      clk,
        input      rst,

        input                    trig,
        input  [WIDTH*DEPTH-1:0] in,

        output                   ovalid,
        output [WIDTH-1:0]       out
    );

    localparam ADDR_WIDTH = $clog2(LENGTH);

    localparam IDLE  = 2'b00;
    localparam CALC  = 2'b01;
    localparam DONE  = 2'b10;

    //define the time counter
    reg ivalid;
    reg [1:0] state;
    reg [ADDR_WIDTH-1:0]  addr;
    reg [WIDTH-1:0] C [7:0];
    reg [WIDTH*DEPTH-1:0] idata;

    wire yvalid;
    wire [47:0]	data;
    wire [WIDTH-1:0] Y;
    wire [WIDTH-1:0] value;

    always@(posedge clk or posedge rst) begin
        if (rst) begin
            addr <= 0;
            state <= 0;
            idata <= 0;
            ivalid <= 0;
        end
        else begin
            case (state)
                IDLE : begin
                    if (trig) begin
                        state <= CALC;
                        idata <= in;
                    end
                end

                CALC : begin
                    ivalid <= 1'b1;
                    if (addr == (LENGTH-1)) begin
                        addr <= 0;
                        state <= DONE;
                    end
                    else begin
                        addr <= addr + 1'b1;
                    end
                end

                DONE : begin
                    idata <= 0;
                    state <= IDLE;
                    ivalid <= 1'b0;
                end

                default : begin
                    ivalid <= 1'b0;
                    state <= IDLE;
                end
            endcase
        end
    end

    gemm1L u_gemm1L(
               //ports
               .clk        ( clk       ),
               .rst        ( rst       ),
               .addr 		( addr 		),
               .data 		( data 		)
           );

    vector #(
               .K     		( 2  		),
               .WIDTH 		( 16 		))
           u_vector(
               //ports
               .clk     ( clk       ),
               .rst     ( rst       ),

               .avalid 	( ivalid    ),
               .A      	( idata     ),

               .bvalid 	( ivalid    ),
               .B      	( data[2*WIDTH-1:0] ),

               .yvalid 	( yvalid  ),
               .Y      	( Y       )
           );

    integer i;
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for (i = 0; i<8; i=i+1) begin
                C[i] <= 0;
            end
        end
        else begin
            C[0] <= data[3*WIDTH-1:2*WIDTH];
            for (i = 0; i<7; i=i+1) begin
                C[i+1] <= C[i];
            end
        end
    end

        tadd u_add(
            .aclk(clk),
            .aresetn(rst),
     
            .s_axis_a_tvalid(yvalid),
            .s_axis_a_tdata(Y),
     
            .s_axis_b_tvalid(yvalid),
            .s_axis_b_tdata(C[7]),
     
            .m_axis_result_tvalid(ovalid),
            .m_axis_result_tdata(value)
        );

    //assign out = value;
    assign out = value[WIDTH-1] ? 0 : value;

endmodule  //GEMMR1
