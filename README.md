# Modular AI Accelerator Pipeline Simulator

## Overview

This project simulates a modular AI accelerator pipeline using Python. It consists of chiplets representing key AI accelerator components such as convolution, activation (ReLU), and matrix multiplication. The pipeline simulator integrates these chiplets, demonstrating their combined operation and latency profiling.

The project includes unit tests for each chiplet to ensure correctness and a lightweight dark-themed Streamlit dashboard for easy interaction and visualization.

---

## Features

- **ConvChiplet**: Simulates a convolution operation with a configurable kernel size and pipeline stages.
- **ActivationChiplet**: Implements the ReLU activation function.
- **MatMulChiplet**: Performs matrix multiplication on 2D lists or NumPy arrays.
- **PipelineSimulator**: Runs the chiplets in sequence and profiles the time taken by each stage.
- **Unit Tests**: Test cases for validating functionality and output correctness.
- **Streamlit Dashboard**: Interactive and dark-themed UI to visualize inputs, outputs, and performance.

---

## Technologies and Frameworks Used

- **Python 3.10+**: Core programming language.
- **NumPy**: Efficient numerical computation and matrix operations.
- **Streamlit**: For building a simple, lightweight, dark-themed web dashboard.
- **unittest**: Python’s built-in unit testing framework.
  
---

## Project Structure

modular_ai_accelerator/
│
├── chiplets/
│ ├── conv_chiplet.py
│ ├── activation_chiplet.py
│ └── matmul_chiplet.py
│
├── python_sim/
│ └── simulate_pipeline.py
│
├── tests/
│ └── test_pipeline_latency.py
│
└── app.py

- `chiplets/` contains individual chiplet implementations.
- `python_sim/` contains the pipeline simulator script.
- `tests/` includes unit tests for the chiplets.
- `app.py` is the Streamlit dashboard interface.

---

## How to Run

### 1. Setup environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

pip install numpy streamlit
