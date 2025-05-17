import streamlit as st
import numpy as np
import sys
import os

# Set up path to import chiplets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'modular_ai_accelerator')))

from chiplets.conv_chiplet import ConvChiplet
from chiplets.activation_chiplet import ActivationChiplet
from chiplets.matmul_chiplet import MatMulChiplet

# --------------------------
# Custom Style: Black Theme + Pulse Line
# --------------------------
st.markdown("""
<style>
body {
    background-color: #000000;
    color: #FFD700;
}

div.stApp {
    background-color: #000000;
    color: #FFD700;
}

.pulse-line {
    width: 100%;
    height: 70px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
    background-color: #000000;
}

.pulse-animation {
    position: absolute;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg,
        transparent 0%,
        #FF0000 20%,
        #FFD700 40%,
        #FFA500 60%,
        #FF0000 80%,
        transparent 100%);
    animation: moveLine 2.5s linear infinite;
    opacity: 0.8;
}

@keyframes moveLine {
    0% {
        transform: translateX(-100%);
    }
    100% {
        transform: translateX(100%);
    }
}
</style>

<div class="pulse-line">
    <div class="pulse-animation"></div>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Streamlit App Content
# --------------------------

st.title("🧠 Modular AI Accelerator Pipeline")

st.sidebar.header("Configuration Settings")

input_size = st.sidebar.slider("Input Matrix Size", 3, 10, 5)
kernel_size = st.sidebar.slider("Kernel Size (Conv)", 2, 5, 3)

# Create input and weight matrices
input_matrix = np.arange(1, input_size * input_size + 1).reshape(input_size, input_size)
matmul_weights = np.ones((input_size - kernel_size + 1, 1))

def run_pipeline(input_matrix, kernel_size, matmul_weights):
    conv = ConvChiplet(kernel_size=kernel_size)
    activation = ActivationChiplet()
    matmul = MatMulChiplet(weights=matmul_weights)

    conv_out = conv.apply(input_matrix)
    act_out = activation.relu(conv_out)
    matmul_out = matmul.multiply(act_out)
    return conv_out, act_out, matmul_out

# Show input matrix
st.subheader("🔢 Input Matrix")
st.dataframe(input_matrix)

if st.button("▶️ Run Pipeline"):
    conv_out, act_out, matmul_out = run_pipeline(input_matrix, kernel_size, matmul_weights)

    st.subheader("🧩 Convolution Output")
    st.dataframe(conv_out)

    st.subheader("🔄 Activation Output (ReLU)")
    st.dataframe(act_out)

    st.subheader("🧮 MatMul Output")
    st.dataframe(matmul_out)
