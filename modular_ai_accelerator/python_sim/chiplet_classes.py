# python_sim/chiplet_classes.py
import numpy as np

class ConvChiplet:
    def forward(self, x):
        return x + 1

class ActivationChiplet:
    def forward(self, x):
        return np.maximum(0, x)

class MatMulChiplet:
    def forward(self, x, weight=2):
        return x * weight
