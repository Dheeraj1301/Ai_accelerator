# python_sim/simulate_pipeline.py
import numpy as np
from chiplet_classes import ConvChiplet, ActivationChiplet, MatMulChiplet

def simulate_pipeline(x):
    conv = ConvChiplet()
    act = ActivationChiplet()
    mat = MatMulChiplet()

    out1 = conv.forward(x)
    out2 = act.forward(out1)
    out3 = mat.forward(out2)
    return out3
