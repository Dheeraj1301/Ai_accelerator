import numpy as np

class ActivationChiplet:
    def relu(self, x):
        x = np.array(x)
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be convertible to a NumPy array")
        return np.maximum(0, x)
