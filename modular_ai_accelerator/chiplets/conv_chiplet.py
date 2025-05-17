class ConvChiplet:
    def __init__(self, kernel=[[1, 0], [0, -1]]):
        self.kernel = kernel

    def process(self, input_matrix):
        # Simple 2x2 convolution without padding/stride for example
        output = []
        rows = len(input_matrix) - 1
        cols = len(input_matrix[0]) - 1
        for i in range(rows):
            row_out = []
            for j in range(cols):
                val = 0
                for ki in range(2):
                    for kj in range(2):
                        val += input_matrix[i + ki][j + kj] * self.kernel[ki][kj]
                row_out.append(val)
            output.append(row_out)
        return output
