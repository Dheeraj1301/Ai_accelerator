# python_sim/visualize.py
import matplotlib.pyplot as plt
import numpy as np
from simulate_pipeline import simulate_pipeline

def visualize_output(input_data):
    output = simulate_pipeline(input_data)
    plt.plot(input_data, label='Input')
    plt.plot(output, label='Output')
    plt.legend()
    plt.title("Chiplet Simulation")
    plt.show()
