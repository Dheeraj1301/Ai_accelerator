import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from chiplets.conv_chiplet import ConvChiplet
from chiplets.activation_chiplet import ActivationChiplet
from chiplets.matmul_chiplet import MatMulChiplet
import time

class PipelineSimulator:
    def __init__(self, input_matrix, kernel_size=3, matmul_weights=None):
        self.input_matrix = np.array(input_matrix)
        self.kernel_size = kernel_size
        self.conv = ConvChiplet(kernel_size=kernel_size)
        self.activation = ActivationChiplet()
        self.matmul = MatMulChiplet(weights=matmul_weights)
        
    def run(self):
        start_time = time.time()
        conv_out = self.conv.apply(self.input_matrix)
        conv_time = time.time() - start_time
        
        start_time = time.time()
        act_out = self.activation.relu(conv_out)
        act_time = time.time() - start_time
        
        start_time = time.time()
        matmul_out = self.matmul.multiply(act_out)
        matmul_time = time.time() - start_time
        
        print("Input Matrix:")
        print(self.input_matrix)
        print("\nConvChiplet Output:")
        print(conv_out)
        print("\nActivationChiplet Output (ReLU):")
        print(act_out)
        print("\nMatMulChiplet Output:")
        print(matmul_out)
        print("\nEnergy Profiling (seconds simulated):")
        print(f"ConvChiplet: {conv_time:.6f}s")
        print(f"ActivationChiplet: {act_time:.6f}s")
        print(f"MatMulChiplet: {matmul_time:.6f}s")

if __name__ == "__main__":
    # 5x5 input matrix for better simulation
    input_matrix = np.arange(1, 26).reshape(5, 5)
    kernel_size = 3
    # MatMul weights with dynamic size to match conv output
    matmul_weights = np.ones((5 - kernel_size + 1, 1))
    
    sim = PipelineSimulator(input_matrix, kernel_size=kernel_size, matmul_weights=matmul_weights)
    sim.run()
