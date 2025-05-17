import numpy as np

class ActivationChiplet:
    def relu(self, x):
        x = np.array(x)
        assert isinstance(x, np.ndarray), "Input must be a NumPy array"
        return np.maximum(0, x)
