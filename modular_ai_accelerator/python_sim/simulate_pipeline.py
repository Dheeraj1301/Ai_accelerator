from chiplets.conv_chiplet import ConvChiplet
from chiplets.activation_chiplet import ActivationChiplet
from chiplets.matmul_chiplet import MatMulChiplet

def simulate_pipeline():
    # Sample input matrix
    input_matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    print("Input Matrix:")
    for row in input_matrix:
        print(row)

    # Convolution
    conv = ConvChiplet()
    conv_output = conv.process(input_matrix)
    print("\nConvChiplet Output:")
    for row in conv_output:
        print(row)

    # Activation
    activation = ActivationChiplet(activation_type="relu")
    activated_output = activation.process(conv_output)
    print("\nActivationChiplet Output (ReLU):")
    for row in activated_output:
        print(row)

    # Matrix multiplication example: multiply activated output by [[1], [1]] to reduce columns
    matmul = MatMulChiplet()
    matrix_b = [[1], [1]]  # Dimensions should match activated_output cols (2 cols) x 1 col
    matmul_output = matmul.process(activated_output, matrix_b)
    print("\nMatMulChiplet Output:")
    for row in matmul_output:
        print(row)

def main():
    simulate_pipeline()

if __name__ == "__main__":
    main()
