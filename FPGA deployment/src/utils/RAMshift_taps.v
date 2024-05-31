`timescale 1 ns / 1 ns

module RAMshift_taps #(
	parameter DELAY_LEN  = 1024,
	parameter DATA_WIDTH = 24
) (
	input	  					  clock,
	input	  					  clken, 
	input	[DATA_WIDTH - 1 : 0]  shiftin,
	output  [DATA_WIDTH - 1 : 0]  shiftout
);

localparam RAM_Length = DELAY_LEN-1;
reg  [$clog2(RAM_Length) - 1 : 0] addr = 0;
reg  [DATA_WIDTH - 1 : 0] shift_ram[RAM_Length - 1 : 0];

// init this shift_ram
integer m;
initial begin
	for (m = 0; m < RAM_Length; m = m + 1) begin
		shift_ram[m] = 0;
	end    
end

always @(posedge clock) begin
    if (clken) begin
        if (addr == (RAM_Length - 1)) 
            addr <= 0;
        else
            addr <= addr + 1;
    end
end

reg	 [DATA_WIDTH - 1 : 0] shiftbuffer = 0;
reg	 [DATA_WIDTH - 1 : 0] shiftout_r = 0;
always @(posedge clock) begin	
	if (clken) begin
        shiftbuffer <= shift_ram[addr];
		shift_ram[addr] <= shiftin;
	end	
    shiftout_r <= shiftbuffer;
end

assign shiftout = shiftout_r;

endmodule

