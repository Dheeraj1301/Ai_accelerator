import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import time

class ConvChiplet:
    def __init__(self, kernel_size=2, bit_width=8):
        self.kernel_size = kernel_size
        self.bit_width = bit_width
        self.kernel = [[-1 for _ in range(kernel_size)] for _ in range(kernel_size)]

    def convolve(self, matrix, enable=True):
        start = time.time()
        if not enable:
            print("ConvChiplet is gated (disabled).")
            return [[0 for _ in range(len(matrix[0]) - self.kernel_size + 1)] for _ in range(len(matrix) - self.kernel_size + 1)], 0.0

        output = []
        for i in range(len(matrix) - self.kernel_size + 1):
            row = []
            for j in range(len(matrix[0]) - self.kernel_size + 1):
                val = 0
                for m in range(self.kernel_size):
                    for n in range(self.kernel_size):
                        val += matrix[i + m][j + n] * self.kernel[m][n]
                row.append(val)
            output.append(row)
        energy = time.time() - start
        return output, energy

class ActivationChiplet:
    def __init__(self, bit_width=8):
        self.bit_width = bit_width

    def relu(self, matrix, enable=True):
        start = time.time()
        if not enable:
            print("ActivationChiplet is gated (disabled).")
            return [[0 for _ in row] for row in matrix], 0.0

        result = [[max(0, val) for val in row] for row in matrix]
        energy = time.time() - start
        return result, energy

class MatMulChiplet:
    def __init__(self, bit_width=8):
        self.bit_width = bit_width

    def matmul(self, matrix_a, matrix_b, enable=True):
        start = time.time()
        if not enable:
            print("MatMulChiplet is gated (disabled).")
            return [[0] * len(matrix_b[0]) for _ in range(len(matrix_a))], 0.0

        result = []
        for i in range(len(matrix_a)):
            row = []
            for j in range(len(matrix_b[0])):
                sum_val = 0
                for k in range(len(matrix_b)):
                    sum_val += matrix_a[i][k] * matrix_b[k][j]
                row.append(sum_val)
            result.append(row)
        energy = time.time() - start
        return result, energy

def simulate_pipeline(config):
    input_matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print("Input Matrix:")
    for row in input_matrix:
        print(row)

    conv_enable = config.get("conv_enable", True)
    act_enable = config.get("act_enable", True)
    matmul_enable = config.get("matmul_enable", True)

    conv = ConvChiplet(kernel_size=config.get("conv_kernel_size", 2), bit_width=config.get("bit_width", 8))
    conv_output, conv_energy = conv.convolve(input_matrix, enable=conv_enable)

    print("\nConvChiplet Output:")
    for row in conv_output:
        print(row)

    act = ActivationChiplet(bit_width=config.get("bit_width", 8))
    activated, act_energy = act.relu(conv_output, enable=act_enable)

    print("\nActivationChiplet Output (ReLU):")
    for row in activated:
        print(row)

    matmul = MatMulChiplet(bit_width=config.get("bit_width", 8))
    dummy_weights = [[1], [1]]
    output, matmul_energy = matmul.matmul(activated, dummy_weights, enable=matmul_enable)

    print("\nMatMulChiplet Output:")
    for row in output:
        print(row)

    print("\nEnergy Profiling (seconds simulated):")
    print(f"ConvChiplet: {conv_energy:.6f}s")
    print(f"ActivationChiplet: {act_energy:.6f}s")
    print(f"MatMulChiplet: {matmul_energy:.6f}s")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate AI Accelerator Pipeline with Configurable Chiplet Enables")
    parser.add_argument("--config", type=str, default="pipeline_config.json", help="Path to the JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)
    simulate_pipeline(config)
from chiplets.conv_chiplet import ConvChiplet
from chiplets.activation_chiplet import ActivationChiplet
from chiplets.matmul_chiplet import MatMulChiplet
import random
import csv

def generate_matrix(rows, cols, low=1, high=10):
    return [[random.randint(low, high) for _ in range(cols)] for _ in range(rows)]

def log_to_file(filename, input_matrix, conv_out, relu_out, matmul_out, timings):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Input Matrix"])
        writer.writerows(input_matrix)
        writer.writerow([])
        writer.writerow(["ConvChiplet Output"])
        writer.writerows(conv_out)
        writer.writerow([])
        writer.writerow(["ActivationChiplet Output (ReLU)"])
        writer.writerows(relu_out)
        writer.writerow([])
        writer.writerow(["MatMulChiplet Output"])
        writer.writerows(matmul_out)
        writer.writerow([])
        writer.writerow(["Energy Profiling (seconds simulated)"])
        for name, time_sec in timings.items():
            writer.writerow([name, f"{time_sec:.6f}s"])

def main():
    input_matrix = generate_matrix(5, 5)
    print("Input Matrix:")
    for row in input_matrix:
        print(row)

    conv = ConvChiplet(kernel_size=3)
    relu = ActivationChiplet()
    matmul = MatMulChiplet()

    conv_out = conv.apply(input_matrix)
    print("\nConvChiplet Output:")
    for row in conv_out:
        print(row)

    relu_out = relu.relu(conv_out)
    print("\nActivationChiplet Output (ReLU):")
    for row in relu_out:
        print(row)

    matmul_out = matmul.multiply(relu_out, [[1] for _ in range(len(relu_out[0]))])
    print("\nMatMulChiplet Output:")
    for row in matmul_out:
        print(row)

    timings = {
        "ConvChiplet": conv.energy_time,
        "ActivationChiplet": relu.energy_time,
        "MatMulChiplet": matmul.energy_time
    }

    print("\nEnergy Profiling (seconds simulated):")
    for name, time_sec in timings.items():
        print(f"{name}: {time_sec:.6f}s")

    log_to_file("simulation_log.csv", input_matrix, conv_out, relu_out, matmul_out, timings)

if __name__ == "__main__":
    main()
