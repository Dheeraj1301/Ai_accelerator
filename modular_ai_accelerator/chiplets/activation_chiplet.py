class ActivationChiplet:
    def __init__(self, activation_type="relu"):
        self.activation_type = activation_type.lower()

    def process(self, input_matrix):
        def relu(x):
            return max(0, x)

        def sigmoid(x):
            import math
            return 1 / (1 + math.exp(-x))

        output = []
        for row in input_matrix:
            output_row = []
            for val in row:
                if self.activation_type == "relu":
                    output_row.append(relu(val))
                elif self.activation_type == "sigmoid":
                    output_row.append(sigmoid(val))
                else:
                    output_row.append(val)  # default no activation
            output.append(output_row)
        return output
