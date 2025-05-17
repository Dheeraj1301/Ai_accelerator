// hdl/top_module.v
module top_module(input clk, input [7:0] in_data, output [15:0] out_data);
  wire [7:0] conv_out, act_out;

  conv_chiplet u1(.clk(clk), .data_in(in_data), .data_out(conv_out));
  activation_chiplet u2(.clk(clk), .in_data(conv_out), .out_data(act_out));
  matmul_chiplet u3(.clk(clk), .a(act_out), .b(8'd2), .result(out_data));
endmodule
