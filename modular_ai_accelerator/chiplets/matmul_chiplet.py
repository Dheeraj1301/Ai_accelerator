import time

class MatMulChiplet:
    def __init__(self, enable=True):
        self.enable = enable
        self.energy_time = 0

    def multiply(self, A, B):
        if not self.enable:
            print("[MatMulChiplet] Disabled")
            return A
        start = time.time()
        result = [[sum(a*b for a,b in zip(row,col)) for col in zip(*B)] for row in A]
        end = time.time()
        self.energy_time = end - start
        return result
