class MatMulChiplet:
    def multiply(self, A, B):
        # A and B are 2D lists (matrices)
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
