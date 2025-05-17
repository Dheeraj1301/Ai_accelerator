import time

class ConvChiplet:
    def __init__(self, kernel_size=3, enable=True):
        self.kernel_size = kernel_size
        self.kernel = [[1 for _ in range(kernel_size)] for _ in range(kernel_size)]
        self.enable = enable
        self.energy_time = 0

    def apply(self, matrix):
        if not self.enable:
            print("[ConvChiplet] Disabled")
            return matrix
        start = time.time()
        output = []
        for i in range(len(matrix) - self.kernel_size + 1):
            row = []
            for j in range(len(matrix[0]) - self.kernel_size + 1):
                val = 0
                for ki in range(self.kernel_size):
                    for kj in range(self.kernel_size):
                        val += matrix[i+ki][j+kj] * self.kernel[ki][kj]
                row.append(val)
            output.append(row)
        end = time.time()
        self.energy_time = end - start
        return output
