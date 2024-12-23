`timescale 1ns/1ns
module	i2c_16b #(
    parameter	CLK_FREQ	=	100_000000,	//100 MHz
    parameter	I2C_FREQ	=	400_000		//10 KHz(< 400KHz)
) (
    //global clock
    input				clk,		//100MHz
    input				rst_n,		//system reset

    //i2c interface
    output				i2c_sclk,	//i2c clock
    inout				i2c_sdat,	//i2c data for bidirection

    //user interface
    input		[8:0]	i2c_config_size,	//i2c config data counte
    output	reg	[8:0]	i2c_config_index,	//i2c config reg index, read 2 reg and write xx reg
    input		[31:0]	i2c_config_data,	//i2c config data
    output				i2c_config_done,	//i2c config timing complete
    output	reg	[15:0]	i2c_rdata			//i2c register data while read i2c slave
);

//----------------------------------------
//Delay xxus until i2c slave is steady
reg	[19:0]	delay_cnt;
localparam	DELAY_TOP = CLK_FREQ/1000;	//1ms Setting time after software/hardware reset
//localparam	DELAY_TOP = 17'hff;			//Just for test
always@(posedge clk or negedge rst_n) begin
    if(!rst_n)
        delay_cnt <= 0;
    else if(delay_cnt < DELAY_TOP)
        delay_cnt <= delay_cnt + 1'b1;
    else
        delay_cnt <= delay_cnt;
end
wire	delay_done = (delay_cnt == DELAY_TOP) ? 1'b1 : 1'b0;	//81us delay


//----------------------------------------
//I2C Control Clock generate
reg	[15:0]	clk_cnt;	//divide for i2c clock
/******************************************
             _______		  _______
SCLK	____|		|________|		 |
         ________________ ______________
SDAT	|________________|______________
         _	              _
CLK_EN	| |______________| |____________
                _			  	 _
CAP_EN	_______| |______________| |_____
*******************************************/
reg	i2c_ctrl_clk;		//i2c control clock, H: valid; L: valid
reg	i2c_transfer_en;	//send i2c data	before, make sure that sdat is steady when i2c_sclk is valid
reg	i2c_capture_en;		//capture i2c data	while sdat is steady from cmos
always@(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        clk_cnt <= 0;
        i2c_ctrl_clk <= 0;
        i2c_transfer_en <= 0;
        i2c_capture_en <= 0;
    end
    else if(delay_done) begin
        if(clk_cnt < (CLK_FREQ/I2C_FREQ) - 1'b1)
            clk_cnt <= clk_cnt + 1'd1;
        else
            clk_cnt <= 0;
        //i2c control clock, H: valid; L: valid
        i2c_ctrl_clk <= ((clk_cnt >= (CLK_FREQ/I2C_FREQ)/4 + 1'b1) &&
                            (clk_cnt < (3*CLK_FREQ/I2C_FREQ)/4 + 1'b1)) ? 1'b1 : 1'b0;
        //send i2c data	before, make sure that sdat is steady when i2c_sclk is valid
        i2c_transfer_en <= (clk_cnt == 16'd0) ? 1'b1 : 1'b0;
        //capture i2c data	while sdat is steady from cmos
        i2c_capture_en <= (clk_cnt == (2*CLK_FREQ/I2C_FREQ)/4 - 1'b1) ? 1'b1 : 1'b0;
    end
    else begin
        clk_cnt <= 0;
        i2c_ctrl_clk <= 0;
        i2c_transfer_en <= 0;
        i2c_capture_en <= 0;
    end
end

//-----------------------------------------
//I2C Timing state Parameter
localparam	I2C_IDLE		=	5'd0;
//Write I2C: {ID_Address, REG_Address, W_REG_Data}
localparam	I2C_WR_START	=	5'd1;
localparam	I2C_WR_IDADDR	=	5'd2;
localparam	I2C_WR_ACK1		=	5'd3;
localparam	I2C_WR_REGADDR	=	5'd4;
localparam	I2C_WR_ACK2	    =	5'd5;
localparam	I2C_WR_REGDATA1	=	5'd6;
localparam	I2C_WR_ACK3		=	5'd7;
localparam	I2C_WR_REGDATA2	=	5'd8;	//16Bit Mode
localparam	I2C_WR_ACK4		=	5'd9;	//16Bit Mode
localparam	I2C_WR_STOP		=	5'd10;
//I2C Read: {ID_Address + REG_Address} + {ID_Address + R_REG_Data}
localparam	I2C_RD_START1	=	5'd11;
localparam	I2C_RD_IDADDR1	=	5'd12;
localparam	I2C_RD_ACK1		=	5'd13;
localparam	I2C_RD_REGADDR	=	5'd14;
localparam	I2C_RD_ACK2		=	5'd15;
localparam	I2C_RD_STOP1	=	5'd16;
localparam	I2C_RD_IDLE		=	5'd17;
localparam	I2C_RD_START2	=	5'd18;
localparam	I2C_RD_IDADDR2	=	5'd19;
localparam	I2C_RD_ACK3		=	5'd20;
localparam	I2C_RD_REGDATA1	=	5'd21;
localparam	I2C_RD_NACK1	=	5'd22;	//16Bit Mode--Read
localparam	I2C_RD_REGDATA2	=	5'd23;	//16Bit Mode--Read
localparam	I2C_RD_NACK2	=	5'd24;
localparam	I2C_RD_STOP2	=	5'd25;


//-----------------------------------------
// FSM: always1
reg	[4:0]	current_state, next_state; //i2c write and read state
always@(posedge clk or negedge rst_n) begin
    if(!rst_n)
        current_state <= I2C_IDLE;
    else if(i2c_transfer_en)
        current_state <= next_state;
end

//-----------------------------------------
wire	i2c_transfer_end = (current_state == I2C_WR_STOP || current_state == I2C_RD_STOP2) ? 1'b1 : 1'b0;
reg		i2c_ack;	//i2c slave renpose successed
always@(posedge clk or negedge rst_n) begin
    if(!rst_n)
        i2c_config_index <= 2;
    else if(i2c_transfer_en) begin
        if(i2c_transfer_end) begin
            if(i2c_config_index < i2c_config_size)
                i2c_config_index <= i2c_config_index + 1'b1;
            else
                i2c_config_index <= i2c_config_size;
        end
        else
            i2c_config_index <= i2c_config_index;
    end
    else
        i2c_config_index <= i2c_config_index;
end
assign	i2c_config_done = (i2c_config_index == i2c_config_size) ? 1'b1 : 1'b0;


//-----------------------------------------
// FSM: always2
reg	[3:0]	i2c_stream_cnt;	//i2c data bit stream count
always@(*) begin
    next_state = I2C_IDLE; 	//state initialization
    case(current_state)
        I2C_IDLE: begin	//5'd0
            if(delay_done) begin	//1ms Setting time after software/hardware reset
                if(i2c_transfer_en) begin
                    if(i2c_config_index < 8'd2)
                        next_state = I2C_RD_START1;	//Read I2C Slave ID
                    else if(i2c_config_index < i2c_config_size)
                        next_state = I2C_WR_START;	//Write Data to I2C
                    else// if(i2c_config_index >= i2c_config_size)
                        next_state = I2C_IDLE;		//Config I2C Complete
                end
                else
                    next_state = next_state;
            end
            else
                next_state = I2C_IDLE;		//Wait I2C Bus is steady
        end
        //Write I2C: {ID_Address, REG_Address, W_REG_Data}
        I2C_WR_START: begin
            if(i2c_transfer_en)
                next_state = I2C_WR_IDADDR;
            else
                next_state = I2C_WR_START;
        end
        I2C_WR_IDADDR:
            if(i2c_transfer_en == 1'b1 && i2c_stream_cnt == 4'd8)
                next_state = I2C_WR_ACK1;
            else
                next_state = I2C_WR_IDADDR;
        I2C_WR_ACK1:
            if(i2c_transfer_en)
                next_state = I2C_WR_REGADDR;
            else
                next_state = I2C_WR_ACK1;
        I2C_WR_REGADDR:
            if(i2c_transfer_en == 1'b1 && i2c_stream_cnt == 4'd8)
                next_state = I2C_WR_ACK2;
            else
                next_state = I2C_WR_REGADDR;
        I2C_WR_ACK2:
            if(i2c_transfer_en)
                next_state = I2C_WR_REGDATA1;
            else
                next_state = I2C_WR_ACK2;
        I2C_WR_REGDATA1:
            if(i2c_transfer_en == 1'b1 && i2c_stream_cnt == 4'd8)
                next_state = I2C_WR_ACK3;
            else
                next_state = I2C_WR_REGDATA1;
        I2C_WR_ACK3:
            if(i2c_transfer_en)
                next_state = I2C_WR_REGDATA2;
            else
                next_state = I2C_WR_ACK3;
        I2C_WR_REGDATA2:
            if(i2c_transfer_en == 1'b1 && i2c_stream_cnt == 4'd8)
                next_state = I2C_WR_ACK4;
            else
                next_state = I2C_WR_REGDATA2;
        I2C_WR_ACK4:
            if(i2c_transfer_en)
                next_state = I2C_WR_STOP;
            else
                next_state = I2C_WR_ACK4;
        I2C_WR_STOP:
            if(i2c_transfer_en)
                next_state = I2C_IDLE;
            else
                next_state = I2C_WR_STOP;
        //I2C Read: {ID_Address + REG_Address} + {ID_Address + R_REG_Data}
        I2C_RD_START1:
            if(i2c_transfer_en)
                next_state = I2C_RD_IDADDR1;
            else
                next_state = I2C_RD_START1;
        I2C_RD_IDADDR1:
            if(i2c_transfer_en == 1'b1 && i2c_stream_cnt == 4'd8)
                next_state = I2C_RD_ACK1;
            else
                next_state = I2C_RD_IDADDR1;
        I2C_RD_ACK1:
            if(i2c_transfer_en)
                next_state = I2C_RD_REGADDR;
            else
                next_state = I2C_RD_ACK1;
        I2C_RD_REGADDR:
            if(i2c_transfer_en == 1'b1 && i2c_stream_cnt == 4'd8)
                next_state = I2C_RD_ACK2;
            else
                next_state = I2C_RD_REGADDR;
        I2C_RD_ACK2:
            if(i2c_transfer_en)
                next_state = I2C_RD_STOP1;
            else
                next_state = I2C_RD_ACK2;
        I2C_RD_STOP1:
            if(i2c_transfer_en)
                next_state = I2C_RD_IDLE;
            else
                next_state = I2C_RD_STOP1;
        I2C_RD_IDLE:
            if(i2c_transfer_en)
                next_state = I2C_RD_START2;
            else
                next_state = I2C_RD_IDLE;
        I2C_RD_START2:
            if(i2c_transfer_en)
                next_state = I2C_RD_IDADDR2;
            else
                next_state = I2C_RD_START2;
        I2C_RD_IDADDR2:
            if(i2c_transfer_en == 1'b1 && i2c_stream_cnt == 4'd8)
                next_state = I2C_RD_ACK3;
            else
                next_state = I2C_RD_IDADDR2;
        I2C_RD_ACK3:
            if(i2c_transfer_en)
                next_state = I2C_RD_REGDATA1;
            else
                next_state = I2C_RD_ACK3;
        I2C_RD_REGDATA1:
            if(i2c_transfer_en == 1'b1 && i2c_stream_cnt == 4'd8)
                next_state = I2C_RD_NACK1;
            else
                next_state = I2C_RD_REGDATA1;
        I2C_RD_NACK1:
            if(i2c_transfer_en)
                next_state = I2C_RD_REGDATA2;
            else
                next_state = I2C_RD_NACK1;
        I2C_RD_REGDATA2:
            if(i2c_transfer_en == 1'b1 && i2c_stream_cnt == 4'd8)
                next_state = I2C_RD_NACK2;
            else
                next_state = I2C_RD_REGDATA2;
        I2C_RD_NACK2:
            if(i2c_transfer_en)
                next_state = I2C_RD_STOP2;
            else
                next_state = I2C_RD_NACK2;
        I2C_RD_STOP2:
            if(i2c_transfer_en)
                next_state = I2C_IDLE;
            else
                next_state = I2C_RD_STOP2;
        default:
            ;	//default vaule
    endcase
end

//-----------------------------------------
// FSM: always3
//reg	i2c_write_flag, i2c_read_flag;
reg	i2c_sdat_out;		//i2c data output
//reg	[3:0]	i2c_stream_cnt;	//i2c data bit stream count
reg	[7:0]	i2c_wdata;	//i2c data prepared to transfer
always@(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        i2c_sdat_out <= 1'b1;
        i2c_stream_cnt <= 0;
        i2c_wdata <= 0;
    end
    else if(i2c_transfer_en) begin
        case(next_state)
            I2C_IDLE: begin
                i2c_sdat_out <= 1'b1;		//idle state
                i2c_stream_cnt <= 0;
                i2c_wdata <= 0;
            end
            //Write I2C: {ID_Address, REG_Address, W_REG_Data}
            I2C_WR_START: begin
                i2c_sdat_out <= 1'b0;
                i2c_stream_cnt <= 0;
                i2c_wdata <= i2c_config_data[31:24];	//ID_Address
            end
            I2C_WR_IDADDR: begin
                i2c_stream_cnt <= i2c_stream_cnt + 1'b1;
                i2c_sdat_out <= i2c_wdata[3'd7 - i2c_stream_cnt];
            end
            I2C_WR_ACK1: begin
                i2c_stream_cnt <= 0;
                i2c_wdata <= i2c_config_data[23:16];		//REG_Address
            end
            I2C_WR_REGADDR: begin
                i2c_stream_cnt <= i2c_stream_cnt + 1'b1;
                i2c_sdat_out <= i2c_wdata[3'd7 - i2c_stream_cnt];
            end
            I2C_WR_ACK2: begin
                i2c_stream_cnt <= 0;
                i2c_wdata <= i2c_config_data[15:8];		//W_REG_Data1
            end
            I2C_WR_REGDATA1: begin
                i2c_stream_cnt <= i2c_stream_cnt + 1'b1;
                i2c_sdat_out <= i2c_wdata[3'd7 - i2c_stream_cnt];
            end
            I2C_WR_ACK3: begin
                i2c_stream_cnt <= 0;
                i2c_wdata <= i2c_config_data[7:0];		//W_REG_Data2
            end
            I2C_WR_REGDATA2: begin
                i2c_stream_cnt <= i2c_stream_cnt + 1'b1;
                i2c_sdat_out <= i2c_wdata[3'd7 - i2c_stream_cnt];
            end
            I2C_WR_ACK4:
                i2c_stream_cnt <= 0;
            I2C_WR_STOP:
                i2c_sdat_out <= 1'b0;
            //I2C Read: {ID_Address + REG_Address} + {ID_Address + R_REG_Data}
            I2C_RD_START1: begin
                i2c_sdat_out <= 1'b0;
                i2c_stream_cnt <= 0;
                i2c_wdata <= i2c_config_data[31:24];
            end
            I2C_RD_IDADDR1: begin
                i2c_stream_cnt <= i2c_stream_cnt + 1'b1;
                i2c_sdat_out <= i2c_wdata[3'd7 - i2c_stream_cnt];
            end
            I2C_RD_ACK1: begin
                i2c_stream_cnt <= 0;
                i2c_wdata <= i2c_config_data[23:16];
            end
            I2C_RD_REGADDR: begin
                i2c_stream_cnt <= i2c_stream_cnt + 1'b1;
                i2c_sdat_out <= i2c_wdata[3'd7 - i2c_stream_cnt];
            end
            I2C_RD_ACK2:
                i2c_stream_cnt <= 0;
            I2C_RD_STOP1:
                i2c_sdat_out <= 1'b0;
            I2C_RD_IDLE:
                i2c_sdat_out <= 1'b1;		//idle state
            //-------------------------
            I2C_RD_START2: begin
                i2c_sdat_out <= 1'b0;
                i2c_stream_cnt <= 0;
                i2c_wdata <= i2c_config_data[31:24];
            end
            I2C_RD_IDADDR2: begin
                i2c_stream_cnt <= i2c_stream_cnt + 1'b1;
                if(i2c_stream_cnt < 5'd7)
                    i2c_sdat_out <= i2c_wdata[3'd7 - i2c_stream_cnt];
                else
                    i2c_sdat_out <= 1'b1;	//Read flag for I2C Timing
            end
            I2C_RD_ACK3:
                i2c_stream_cnt <= 0;
            I2C_RD_REGDATA1:
                i2c_stream_cnt <= i2c_stream_cnt + 1'b1;
            I2C_RD_NACK1: begin
                i2c_stream_cnt <= 0;
                i2c_sdat_out <= 1'b0;	//NACK1, It is very important!
            end
            I2C_RD_REGDATA2:
                i2c_stream_cnt <= i2c_stream_cnt + 1'b1;
            I2C_RD_NACK2:
                i2c_sdat_out <= 1'b1;	//NACK2
            I2C_RD_STOP2:
                i2c_sdat_out <= 1'b0;
        endcase
    end
    else begin
        i2c_stream_cnt <= i2c_stream_cnt;
        i2c_sdat_out <= i2c_sdat_out;
    end
end

//---------------------------------------------
//respone from slave for i2c data transfer
reg	i2c_ack1, i2c_ack2, i2c_ack3, i2c_ack4;
reg	[15:0]	i2c_rdata_r;
always@(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        {i2c_ack1, i2c_ack2, i2c_ack3, i2c_ack4} <= 4'b111;
        i2c_ack <= 1'b1;
        i2c_rdata_r <= 0;
        i2c_rdata <= 0;
    end
    else if(i2c_capture_en) begin
        case(next_state)
            I2C_IDLE: begin
                {i2c_ack1, i2c_ack2, i2c_ack3, i2c_ack4} <= 4'b1111;
                i2c_ack <= 1'b1;
            end
            //Write I2C: {ID_Address, REG_Address, W_REG_Data}
            I2C_WR_ACK1:
                i2c_ack1 <= i2c_sdat;
            I2C_WR_ACK2:
                i2c_ack2 <= i2c_sdat;
            I2C_WR_ACK3:
                i2c_ack3 <= i2c_sdat;
            I2C_WR_ACK4:
                i2c_ack4 <= i2c_sdat;
            I2C_WR_STOP:
                i2c_ack <= (i2c_ack1 | i2c_ack2 | i2c_ack3 | i2c_ack4);
            //I2C Read: {ID_Address + REG_Address} + {ID_Address + R_REG_Data}
            I2C_RD_ACK1:
                i2c_ack1 <= i2c_sdat;
            I2C_RD_ACK2:
                i2c_ack2 <= i2c_sdat;
            I2C_RD_ACK3:
                i2c_ack3 <= i2c_sdat;
            I2C_RD_STOP2:
                i2c_ack <= (i2c_ack1 | i2c_ack2 | i2c_ack3);
            I2C_RD_REGDATA1:
                i2c_rdata_r[15:8] <= {i2c_rdata_r[14:8], i2c_sdat};
            I2C_RD_REGDATA2:
                i2c_rdata_r[7:0]  <= {i2c_rdata_r[6:0], i2c_sdat};
            I2C_RD_NACK2:
                i2c_rdata <= i2c_rdata_r;	//Update iec_rdata;
            default: ;
        endcase
    end
    else begin
        {i2c_ack1, i2c_ack2, i2c_ack3, i2c_ack4} <= {i2c_ack1, i2c_ack2, i2c_ack3, i2c_ack4};
        i2c_ack <= i2c_ack;
        i2c_rdata <= i2c_rdata;
    end
end

//---------------------------------------------------
wire	bir_en =(	current_state == I2C_WR_ACK1 || current_state == I2C_WR_ACK2 || current_state == I2C_WR_ACK3 || current_state == I2C_WR_ACK4 ||
                current_state == I2C_RD_ACK1 || current_state == I2C_RD_ACK2 || current_state == I2C_RD_ACK3 ||
                current_state == I2C_RD_REGDATA1 || current_state == I2C_RD_REGDATA2) ? 1'b1 : 1'b0;
assign	i2c_sclk = (current_state >= I2C_WR_IDADDR && current_state <= I2C_WR_ACK4 ||
                    current_state >= I2C_RD_IDADDR1 && current_state <= I2C_RD_ACK2 ||
                    current_state >= I2C_RD_IDADDR2 && current_state <= I2C_RD_NACK2) ?
        i2c_ctrl_clk : 1'b1;
assign	i2c_sdat = (~bir_en) ? i2c_sdat_out : 1'bz;


endmodule
