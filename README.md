
### AI Accelerator Pipeline Simulator
## Project Overview
This project simulates a modular AI accelerator pipeline designed to mimic core operations in deep learning accelerators. The framework consists of individual chiplets representing fundamental computation units such as matrix multiplication, convolution, and activation functions. By modeling each step separately and then combining them in a pipeline, this simulation provides insights into latency, energy consumption, and computational behavior of AI accelerators.

The project includes a lightweight Streamlit dashboard to visualize inputs, intermediate results, and performance metrics, making it easier to understand the accelerator’s operation in a user-friendly and interactive environment.

## Technologies & Frameworks
Python 3.10+: Core programming language for development

NumPy: Efficient numerical computations and matrix operations

Streamlit: Lightweight and interactive web dashboard for visualization

Unittest: Python’s built-in testing framework for code verification

## How to Run and Test
# Running the Pipeline Simulation
Ensure all dependencies are installed:
  pip install numpy streamlit
Run the simulation script:
  python python_sim/simulate_pipeline.py
Expected output includes input matrix, outputs at each stage, and timing metrics for convolution, activation, and matmul chiplets.
Running Unit Tests
Tests cover core functionalities of chiplets:
  python -m unittest discover -s tests
Using the Streamlit Dashboard
Launch the dashboard to interact with the pipeline visually:
  streamlit run app.py
## The dashboard allows you to:

Upload or generate input matrices dynamically.

Adjust kernel sizes.

Visualize convolution, activation, and matmul outputs.

View simulated latency for each pipeline stage.

## Troubleshooting and Fixes
During development, several issues were resolved to ensure smooth operation and consistency:

MatMulChiplet Method Signature: Fixed method to accept two arguments (multiply(A, B)) and corrected calls to match.

ReLU Activation Comparison: Replaced direct assertEqual on numpy arrays with np.array_equal() in tests to avoid ambiguity errors.

PipelineSimulator Initialization: Removed unnecessary constructor parameters from MatMulChiplet to match the defined class.

Module Import Paths: Adjusted Python path imports to accommodate directory structure, ensuring chiplets package modules are found when app.py runs outside the package.

Consistent Data Types: Converted inputs to NumPy arrays across all modules for reliable numerical operations and test consistency.

Test Execution Output: Added verbosity flags and __main__ guard to test scripts to get detailed output during test runs.

Streamlit Styling: Created a lightweight dark-themed dashboard with minimal dependencies for easy deployment.

## Why This Project Matters for AI Accelerators
AI accelerators are specialized hardware designed to speed up the computation of neural networks, especially matrix multiplications and convolutions, which dominate deep learning workloads. This project:

Models the modular design: Reflects current trends in chiplet-based accelerator design, where separate specialized units (chiplets) are interconnected for flexible, scalable computation.

Simulates pipelining: Helps understand latency and throughput trade-offs in pipelined architectures.

Provides insights for optimization: Timing outputs simulate energy and performance metrics valuable for hardware/software co-design.

Is educational: Serves as a foundation for developers and researchers to prototype new accelerator ideas without physical hardware.

Is extensible: Easily expanded with more chiplets (e.g., pooling, normalization) or connected to hardware emulators.
## File Structure:
  .
├── app.py                 # Streamlit dashboard
├── python_sim/
│   └── simulate_pipeline.py   # Pipeline simulation script
├── chiplets/
│   ├── matmul_chiplet.py       # MatMulChiplet module
│   ├── conv_chiplet.py         # ConvChiplet module
│   └── activation_chiplet.py   # ActivationChiplet module
├── tests/
│   └── test_pipeline_latency.py    # Unit tests for chiplets
└── README.md               # This file
## Example Prompts for Testing and Running
Run simulation with default settings:
  python python_sim/simulate_pipeline.py
Run unit tests with detailed output:
  python -m unittest -v tests/test_pipeline_latency.py
Launch interactive dashboard:
  streamlit run app.py
## Final Notes
This modular approach to simulating AI accelerators bridges software and hardware concepts, enabling deeper understanding of how individual computational units impact overall neural network performance. Through ongoing enhancements and community feedback, this project can evolve into a valuable tool for both education and research in AI hardware acceleration.

