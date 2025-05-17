// chiplets/matmul_chiplet.v
module matmul_chiplet(input clk, input [7:0] a, input [7:0] b, output [15:0] result);
  assign result = a * b; // Basic multiplication
endmodule
