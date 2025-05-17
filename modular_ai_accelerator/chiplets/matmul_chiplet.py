class MatMulChiplet:
    def __init__(self, weights=None):
        # Store weights as matrix B for multiplication
        self.weights = weights

    def multiply(self, A, B=None):
        # If B not provided, use self.weights
        if B is None:
            if self.weights is None:
                raise ValueError("No weights provided for multiplication.")
            B = self.weights

        # A and B are 2D lists or np.arrays (matrices)
        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(B[0])):
                s = 0
                for k in range(len(B)):
                    s += A[i][k] * B[k][j]
                row.append(s)
            result.append(row)
        return result
