module topv (input clk, input [15:0] a, input [15:0] b, output reg [31:0] y);
  always @(posedge clk) y <= a * b;
endmodule
