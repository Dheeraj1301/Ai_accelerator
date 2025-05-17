// chiplets/activation_chiplet.v
module activation_chiplet(input clk, input signed [7:0] in_data, output [7:0] out_data);
  assign out_data = (in_data > 0) ? in_data : 0; // ReLU
endmodule
