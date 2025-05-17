import time

class ActivationChiplet:
    def __init__(self, enable=True):
        self.enable = enable
        self.energy_time = 0

    def relu(self, matrix):
        if not self.enable:
            print("[ActivationChiplet] Disabled")
            return matrix
        start = time.time()
        activated = [[max(0, val) for val in row] for row in matrix]
        end = time.time()
        self.energy_time = end - start
        return activated
