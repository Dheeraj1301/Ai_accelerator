import numpy as np

class ConvChiplet:
    def __init__(self, kernel_size=3, pipeline_stages=2):
        self.kernel_size = kernel_size
        self.pipeline_stages = pipeline_stages
        self.kernel = np.ones((kernel_size, kernel_size), dtype=int)

    def apply(self, matrix):
        matrix = np.array(matrix)
        rows, cols = matrix.shape
        out_rows = rows - self.kernel_size + 1
        out_cols = cols - self.kernel_size + 1

        if out_rows <= 0 or out_cols <= 0:
            raise ValueError("Input matrix is smaller than the kernel size.")

        output = np.zeros((out_rows, out_cols), dtype=int)

        # Simple pipeline simulation: split rows among pipeline stages
        for stage in range(self.pipeline_stages):
            start_row = (out_rows * stage) // self.pipeline_stages
            end_row = (out_rows * (stage + 1)) // self.pipeline_stages
            for i in range(start_row, end_row):
                for j in range(out_cols):
                    sub_mat = matrix[i:i+self.kernel_size, j:j+self.kernel_size]
                    output[i, j] = np.sum(sub_mat * self.kernel)

        return output
