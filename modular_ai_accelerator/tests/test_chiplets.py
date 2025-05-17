# tests/test_chiplets.py
import numpy as np
from python_sim.chiplet_classes import ConvChiplet, ActivationChiplet

def test_conv_chiplet():
    c = ConvChiplet()
    assert np.all(c.forward(np.array([1, 2])) == np.array([2, 3]))

def test_activation_chiplet():
    a = ActivationChiplet()
    assert np.all(a.forward(np.array([-1, 2])) == np.array([0, 2]))
