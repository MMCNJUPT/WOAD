
module gemm3C (
    input         addr,
    output [15:0] data
);

reg [15:0] out = 0;
always @(*) begin
    case (addr)
		1'd0 : begin out = 16'b0010100110100110; end // 4.413E-2
		1'd1 : begin out = 16'b1001110111100001; end // -5.74E-3
    endcase
end
assign data = out;

endmodule //gemm3C
