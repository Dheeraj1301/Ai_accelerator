// chiplets/conv_chiplet.v
module conv_chiplet(input clk, input [7:0] data_in, output [7:0] data_out);
  assign data_out = data_in + 1; // Dummy convolution operation
endmodule
