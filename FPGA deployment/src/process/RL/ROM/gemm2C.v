
module gemm2C (
    input         clk,
    input         rst,
    input  [7:0]  addr,
    output [15:0] data
);

reg [15:0] out;
always @(posedge clk or posedge rst) begin
    if (rst) begin
        out <= 0;
    end
    else begin
        case (addr)
            8'd0 : begin out <= 16'b1010010011011101; end
            8'd1 : begin out <= 16'b1001101010011111; end
            8'd2 : begin out <= 16'b1010010111110000; end
            8'd3 : begin out <= 16'b0010100100011011; end
            8'd4 : begin out <= 16'b1010100001110000; end
            8'd5 : begin out <= 16'b0010101010111100; end
            8'd6 : begin out <= 16'b0010101010100100; end
            8'd7 : begin out <= 16'b0010100101000010; end
            8'd8 : begin out <= 16'b1010010011101100; end
            8'd9 : begin out <= 16'b1010010001000101; end
            8'd10 : begin out <= 16'b0010101001001110; end
            8'd11 : begin out <= 16'b0001111001011011; end
            8'd12 : begin out <= 16'b1010011001001001; end
            8'd13 : begin out <= 16'b0010100010110111; end
            8'd14 : begin out <= 16'b0010000101100110; end
            8'd15 : begin out <= 16'b1010100000101100; end
            8'd16 : begin out <= 16'b1010010111011111; end
            8'd17 : begin out <= 16'b0010101100000011; end
            8'd18 : begin out <= 16'b1010100100100011; end
            8'd19 : begin out <= 16'b1010010111001010; end
            8'd20 : begin out <= 16'b0010011110001110; end
            8'd21 : begin out <= 16'b0010100000100100; end
            8'd22 : begin out <= 16'b1010010101111100; end
            8'd23 : begin out <= 16'b1010001110010100; end
            8'd24 : begin out <= 16'b1010010011110000; end
            8'd25 : begin out <= 16'b1010101111110000; end
            8'd26 : begin out <= 16'b1010011101110010; end
            8'd27 : begin out <= 16'b0010101101000001; end
            8'd28 : begin out <= 16'b0010101010000000; end
            8'd29 : begin out <= 16'b0010000101110100; end
            8'd30 : begin out <= 16'b0010001000001101; end
            8'd31 : begin out <= 16'b1010010001100100; end
            8'd32 : begin out <= 16'b0010100001100101; end
            8'd33 : begin out <= 16'b1010100101110011; end
            8'd34 : begin out <= 16'b0010110000001001; end
            8'd35 : begin out <= 16'b0010010110010111; end
            8'd36 : begin out <= 16'b0010000000001110; end
            8'd37 : begin out <= 16'b1010100010100110; end
            8'd38 : begin out <= 16'b1001100110001110; end
            8'd39 : begin out <= 16'b1010101101011001; end
            8'd40 : begin out <= 16'b0010010100101001; end
            8'd41 : begin out <= 16'b0010100100010011; end
            8'd42 : begin out <= 16'b0010101001010011; end
            8'd43 : begin out <= 16'b0010001110110011; end
            8'd44 : begin out <= 16'b1010000010001011; end
            8'd45 : begin out <= 16'b0010100111101100; end
            8'd46 : begin out <= 16'b1001000010101000; end
            8'd47 : begin out <= 16'b0010100011100001; end
            8'd48 : begin out <= 16'b1010101100101100; end
            8'd49 : begin out <= 16'b0010011010000111; end
            8'd50 : begin out <= 16'b0010100110011101; end
            8'd51 : begin out <= 16'b0010101101010001; end
            8'd52 : begin out <= 16'b1010100011111101; end
            8'd53 : begin out <= 16'b0010100110111100; end
            8'd54 : begin out <= 16'b1010100111000111; end
            8'd55 : begin out <= 16'b0010011110110100; end
            8'd56 : begin out <= 16'b0010101001010001; end
            8'd57 : begin out <= 16'b1010010111001000; end
            8'd58 : begin out <= 16'b1010011010111011; end
            8'd59 : begin out <= 16'b0010010011101110; end
            8'd60 : begin out <= 16'b1010101101110100; end
            8'd61 : begin out <= 16'b1001100101000001; end
            8'd62 : begin out <= 16'b0001110110111001; end
            8'd63 : begin out <= 16'b1001101100100001; end
            8'd64 : begin out <= 16'b0010000100110000; end
            8'd65 : begin out <= 16'b0001111001111000; end
            8'd66 : begin out <= 16'b1010101100011111; end
            8'd67 : begin out <= 16'b1010000100001000; end
            8'd68 : begin out <= 16'b1010101101011001; end
            8'd69 : begin out <= 16'b1010100000101010; end
            8'd70 : begin out <= 16'b0010010111010000; end
            8'd71 : begin out <= 16'b0010010001101110; end
            8'd72 : begin out <= 16'b1010010101001010; end
            8'd73 : begin out <= 16'b0010010110100111; end
            8'd74 : begin out <= 16'b0010010001001010; end
            8'd75 : begin out <= 16'b1010101110110010; end
            8'd76 : begin out <= 16'b0010100100101010; end
            8'd77 : begin out <= 16'b1010011000001100; end
            8'd78 : begin out <= 16'b1010000110001010; end
            8'd79 : begin out <= 16'b0001110111101010; end
            8'd80 : begin out <= 16'b0010000110010001; end
            8'd81 : begin out <= 16'b0001010010001100; end
            8'd82 : begin out <= 16'b1010100100111110; end
            8'd83 : begin out <= 16'b0010101011010011; end
            8'd84 : begin out <= 16'b1010101101011000; end
            8'd85 : begin out <= 16'b1001011010011101; end
            8'd86 : begin out <= 16'b0010011001111111; end
            8'd87 : begin out <= 16'b0001101001000111; end
            8'd88 : begin out <= 16'b1010101011101100; end
            8'd89 : begin out <= 16'b0010010010001011; end
            8'd90 : begin out <= 16'b0010001001110110; end
            8'd91 : begin out <= 16'b1010100010001100; end
            8'd92 : begin out <= 16'b0010010010000111; end
            8'd93 : begin out <= 16'b1010100101110010; end
            8'd94 : begin out <= 16'b0010100010111100; end
            8'd95 : begin out <= 16'b0010000101000000; end
            8'd96 : begin out <= 16'b0010100111100011; end
            8'd97 : begin out <= 16'b1010100101010001; end
            8'd98 : begin out <= 16'b1010100100111000; end
            8'd99 : begin out <= 16'b1010011100011000; end
            8'd100 : begin out <= 16'b0010100001100111; end
            8'd101 : begin out <= 16'b1010100100110001; end
            8'd102 : begin out <= 16'b1010101010010001; end
            8'd103 : begin out <= 16'b1010010011000111; end
            8'd104 : begin out <= 16'b0010101010011000; end
            8'd105 : begin out <= 16'b0010011110101101; end
            8'd106 : begin out <= 16'b0010010110011100; end
            8'd107 : begin out <= 16'b0010011000010101; end
            8'd108 : begin out <= 16'b0010100101110010; end
            8'd109 : begin out <= 16'b1010101010000010; end
            8'd110 : begin out <= 16'b1001101111000100; end
            8'd111 : begin out <= 16'b1010101111100011; end
            8'd112 : begin out <= 16'b1010010111101110; end
            8'd113 : begin out <= 16'b1010101011010011; end
            8'd114 : begin out <= 16'b1010011001101001; end
            8'd115 : begin out <= 16'b0001110010100101; end
            8'd116 : begin out <= 16'b0010010110000001; end
            8'd117 : begin out <= 16'b1010010011010001; end
            8'd118 : begin out <= 16'b0010101111001001; end
            8'd119 : begin out <= 16'b0010101100111111; end
            8'd120 : begin out <= 16'b1010001101101000; end
            8'd121 : begin out <= 16'b0001110011001101; end
            8'd122 : begin out <= 16'b1001110000110100; end
            8'd123 : begin out <= 16'b1010011010001001; end
            8'd124 : begin out <= 16'b0010001001111101; end
            8'd125 : begin out <= 16'b0010100010001100; end
            8'd126 : begin out <= 16'b1010101001110011; end
            8'd127 : begin out <= 16'b1010011110011111; end
            8'd128 : begin out <= 16'b1010101010110000; end
            8'd129 : begin out <= 16'b0010010001001010; end
            8'd130 : begin out <= 16'b1010100100100100; end
            8'd131 : begin out <= 16'b1010001110111011; end
            8'd132 : begin out <= 16'b0010100111011000; end
            8'd133 : begin out <= 16'b1010100000110100; end
            8'd134 : begin out <= 16'b1010100010111010; end
            8'd135 : begin out <= 16'b0010011000010110; end
            8'd136 : begin out <= 16'b0010010110011011; end
            8'd137 : begin out <= 16'b1001110110001000; end
            8'd138 : begin out <= 16'b1010101010101000; end
            8'd139 : begin out <= 16'b0010011011111111; end
            8'd140 : begin out <= 16'b0010101000111110; end
            8'd141 : begin out <= 16'b0010101001000011; end
            8'd142 : begin out <= 16'b1010001010011110; end
            8'd143 : begin out <= 16'b1010100011001111; end
            8'd144 : begin out <= 16'b0001110110111100; end
            8'd145 : begin out <= 16'b0010100001011000; end
            8'd146 : begin out <= 16'b1001001011010000; end
            8'd147 : begin out <= 16'b0010001010000101; end
            8'd148 : begin out <= 16'b1010100011001001; end
            8'd149 : begin out <= 16'b0010100011010100; end
            8'd150 : begin out <= 16'b0010011011001001; end
            8'd151 : begin out <= 16'b1010101011001101; end
            8'd152 : begin out <= 16'b1010010110111101; end
            8'd153 : begin out <= 16'b0001100010110010; end
            8'd154 : begin out <= 16'b1010010011010000; end
            8'd155 : begin out <= 16'b1010010010010010; end
            8'd156 : begin out <= 16'b0010010101110111; end
            8'd157 : begin out <= 16'b0010101000111011; end
            8'd158 : begin out <= 16'b0010110000100100; end
            8'd159 : begin out <= 16'b1010011010000100; end
            8'd160 : begin out <= 16'b1010000001110111; end
            8'd161 : begin out <= 16'b0001100100010011; end
            8'd162 : begin out <= 16'b0001111000110000; end
            8'd163 : begin out <= 16'b0010101011101010; end
            8'd164 : begin out <= 16'b0001111101111100; end
            8'd165 : begin out <= 16'b1010100110110001; end
            8'd166 : begin out <= 16'b1010101000001101; end
            8'd167 : begin out <= 16'b0010011000001111; end
            8'd168 : begin out <= 16'b0010011010000110; end
            8'd169 : begin out <= 16'b0010011111110010; end
            8'd170 : begin out <= 16'b1010000101001101; end
            8'd171 : begin out <= 16'b1010100110100110; end
            8'd172 : begin out <= 16'b0010100010001011; end
            8'd173 : begin out <= 16'b1010101010001110; end
            8'd174 : begin out <= 16'b0001110000001111; end
            8'd175 : begin out <= 16'b0010011111111010; end
            8'd176 : begin out <= 16'b1001010001111101; end
            8'd177 : begin out <= 16'b1010100010010001; end
            8'd178 : begin out <= 16'b0001000100110011; end
            8'd179 : begin out <= 16'b0001110101100010; end
            8'd180 : begin out <= 16'b1010000100111100; end
            8'd181 : begin out <= 16'b1010001001001011; end
            8'd182 : begin out <= 16'b0010100000101100; end
            8'd183 : begin out <= 16'b0010100111011110; end
            8'd184 : begin out <= 16'b0010100001011010; end
            8'd185 : begin out <= 16'b0010000111100000; end
            8'd186 : begin out <= 16'b0010011000011101; end
            8'd187 : begin out <= 16'b0010011011110010; end
            8'd188 : begin out <= 16'b0010010000111000; end
            8'd189 : begin out <= 16'b0010010010010000; end
            8'd190 : begin out <= 16'b0010101101111110; end
            8'd191 : begin out <= 16'b1010100110110111; end
            8'd192 : begin out <= 16'b1010010110011010; end
            8'd193 : begin out <= 16'b0010101000010011; end
            8'd194 : begin out <= 16'b0010011110001010; end
            8'd195 : begin out <= 16'b1010011110101111; end
            8'd196 : begin out <= 16'b1010101101010010; end
            8'd197 : begin out <= 16'b0010101100110100; end
            8'd198 : begin out <= 16'b1001100000110100; end
            8'd199 : begin out <= 16'b0010101000000000; end
            8'd200 : begin out <= 16'b0010011011100010; end
            8'd201 : begin out <= 16'b1010101110101011; end
            8'd202 : begin out <= 16'b1010100000010110; end
            8'd203 : begin out <= 16'b1010001011001111; end
            8'd204 : begin out <= 16'b0001110010101000; end
            8'd205 : begin out <= 16'b1010101010010101; end
            8'd206 : begin out <= 16'b1010010110010010; end
            8'd207 : begin out <= 16'b0010010100110001; end
            8'd208 : begin out <= 16'b0010011010000100; end
            8'd209 : begin out <= 16'b1001110100100110; end
            8'd210 : begin out <= 16'b1010100001010000; end
            8'd211 : begin out <= 16'b0010101100101100; end
            8'd212 : begin out <= 16'b1010000110101101; end
            8'd213 : begin out <= 16'b0001101000101100; end
            8'd214 : begin out <= 16'b0010101010010011; end
            8'd215 : begin out <= 16'b1010101011100111; end
            8'd216 : begin out <= 16'b0010011010100111; end
            8'd217 : begin out <= 16'b0010101101000000; end
            8'd218 : begin out <= 16'b0010101101100111; end
            8'd219 : begin out <= 16'b0010010010011001; end
            8'd220 : begin out <= 16'b0010011111000110; end
            8'd221 : begin out <= 16'b1001111111010011; end
            8'd222 : begin out <= 16'b0010010001001001; end
            8'd223 : begin out <= 16'b1010101010111100; end
            8'd224 : begin out <= 16'b1010011111100000; end
            8'd225 : begin out <= 16'b1010001001100011; end
            8'd226 : begin out <= 16'b1010101101010111; end
            8'd227 : begin out <= 16'b1010011110100001; end
            8'd228 : begin out <= 16'b1010100100101000; end
            8'd229 : begin out <= 16'b0010010000100011; end
            8'd230 : begin out <= 16'b0010000000010110; end
            8'd231 : begin out <= 16'b0010101000101000; end
            8'd232 : begin out <= 16'b0001110101101010; end
            8'd233 : begin out <= 16'b1010011111101111; end
            8'd234 : begin out <= 16'b1010101011110110; end
            8'd235 : begin out <= 16'b1010011010110000; end
            8'd236 : begin out <= 16'b0010100110111101; end
            8'd237 : begin out <= 16'b0010100010000111; end
            8'd238 : begin out <= 16'b1010010111111010; end
            8'd239 : begin out <= 16'b1010100111110000; end
            8'd240 : begin out <= 16'b1010100001100010; end
            8'd241 : begin out <= 16'b1010000011111010; end
            8'd242 : begin out <= 16'b1010000000001011; end
            8'd243 : begin out <= 16'b0010011011111100; end
            8'd244 : begin out <= 16'b0010100011011000; end
            8'd245 : begin out <= 16'b1010101010000101; end
            8'd246 : begin out <= 16'b0010101010010100; end
            8'd247 : begin out <= 16'b1010011100110001; end
            8'd248 : begin out <= 16'b0010011000001000; end
            8'd249 : begin out <= 16'b0010010101001011; end
            8'd250 : begin out <= 16'b0010010000100010; end
            8'd251 : begin out <= 16'b1010100001110110; end
            8'd252 : begin out <= 16'b0010100011000011; end
            8'd253 : begin out <= 16'b0001110111110101; end
            8'd254 : begin out <= 16'b1010010010110001; end
            8'd255 : begin out <= 16'b1001110110010000; end
        endcase
    end
end
assign data = out;

endmodule //gemm2C
