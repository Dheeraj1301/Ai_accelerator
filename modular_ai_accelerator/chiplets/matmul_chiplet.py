class MatMulChiplet:
    def process(self, matrix_a, matrix_b):
        # Simple matrix multiplication assuming valid dimensions
        result = []
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        cols_b = len(matrix_b[0])

        for i in range(rows_a):
            row_result = []
            for j in range(cols_b):
                val = 0
                for k in range(cols_a):
                    val += matrix_a[i][k] * matrix_b[k][j]
                row_result.append(val)
            result.append(row_result)
        return result
