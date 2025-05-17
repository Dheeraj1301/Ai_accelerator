# tests/test_chiplets.py
import pytest
import numpy as np
from chiplets.conv_chiplet import ConvChiplet
from chiplets.activation_chiplet import ActivationChiplet
from chiplets.matmul_chiplet import MatMulChiplet

@pytest.fixture
def sample_matrix():
    return np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

@pytest.fixture
def matmul_weights():
    return np.array([[1], [1]])

def test_conv_output_shape(sample_matrix):
    conv = ConvChiplet(kernel_size=3)
    out = conv.apply(sample_matrix)
    assert out.shape == (1, 1)

def test_relu_activation():
    relu = ActivationChiplet()
    result = relu.relu(np.array([[5, -1], [-3, 7]]))
    assert (result == np.array([[5, 0], [0, 7]])).all()

def test_matmul_output(sample_matrix, matmul_weights):
    matmul = MatMulChiplet(weights=matmul_weights)
    A = np.array([[1, 2], [3, 4]])
    result = matmul.multiply(A)
    expected = np.array([[3], [7]])
    assert (result == expected).all()

