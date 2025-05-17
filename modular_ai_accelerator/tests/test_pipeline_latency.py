
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import unittest
from chiplets.conv_chiplet import ConvChiplet
from chiplets.activation_chiplet import ActivationChiplet
from chiplets.matmul_chiplet import MatMulChiplet

class TestChiplets(unittest.TestCase):
    def setUp(self):
        self.matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

    def test_conv_output_shape(self):
        conv = ConvChiplet(kernel_size=3)
        out = conv.apply(self.matrix)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 1)

    def test_relu_activation(self):
        relu = ActivationChiplet()
        result = relu.relu([[5, -1], [-3, 7]])
        self.assertEqual(result, [[5, 0], [0, 7]])

    def test_matmul_output(self):
        matmul = MatMulChiplet()
        A = [[1, 2], [3, 4]]
        B = [[1], [1]]
        result = matmul.multiply(A, B)
        self.assertEqual(result, [[3], [7]])

if __name__ == "__main__":
    unittest.main()
