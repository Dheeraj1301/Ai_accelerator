
# AI Accelerator Pipeline Simulator
## Project Overview

This project simulates a modular AI accelerator pipeline, modeling fundamental computation units (chiplets) commonly found in deep learning hardware accelerators. It includes convolution, activation (ReLU), and matrix multiplication stages connected in a pipeline to analyze latency and energy profiling. The framework also features an interactive **Streamlit dashboard** to visualize computations and performance metrics.

---

## Technologies & Frameworks

- **Python 3.10+** — Core development language  
- **NumPy** — Numerical matrix operations  
- **Streamlit** — Lightweight dashboard for interactive visualization  
- **Unittest** — Python’s unit testing framework  

---

## Code Modules

### MatMulChiplet

Implements matrix multiplication with the method:

```python
multiply(A, B)
```
ConvChiplet
Performs 2D convolution with pipelining simulation:

```python

apply(matrix)
```
ActivationChiplet
Applies ReLU activation:

```python

relu(x)
```
PipelineSimulator
Coordinates the pipeline execution and profiling:

```python

run()
```
How to Run
1. Install Dependencies
```bash

pip install numpy streamlit
```
2. Run Pipeline Simulation
```bash

python python_sim/simulate_pipeline.py
```
3. Run Unit Tests
```bash

python -m unittest discover -s tests -v
```
4. Launch Streamlit Dashboard
```bash

streamlit run app.py
```
### Troubleshooting & Fixes
Fixed method signature mismatch in MatMulChiplet to accept two input matrices.

Used np.array_equal() in tests to avoid assertion errors with NumPy arrays.

Adjusted imports to reflect the project’s directory structure for smooth execution.

Ensured consistent use of NumPy arrays across chiplets for numerical stability.

Added timing simulation in PipelineSimulator for energy profiling insights.

Benefits for AI Accelerator Research
Models chiplet-based modular accelerator designs.

Simulates pipelining stages to estimate latency and throughput.

Offers a testbed for evaluating new accelerator concepts without hardware.

Provides educational insights bridging hardware and software aspects of AI computation.

### Project Structure
```
.
├── app.py
├── python_sim/
│   └── simulate_pipeline.py
├── chiplets/
│   ├── matmul_chiplet.py
│   ├── conv_chiplet.py
│   └── activation_chiplet.py
├── tests/
│   └── test_pipeline_latency.py
└── README.md
```
Example Commands
Run simulation with default parameters:

```bash

python python_sim/simulate_pipeline.py
```
Run detailed tests:

```bash

python -m unittest -v tests/test_pipeline_latency.py
```
Start the interactive dashboard:

```bash

streamlit run app.py
```
### 
Acknowledgements
Developed as part of a modular AI accelerator simulation project. Inspired by modern AI chiplet architectures and the need for accessible accelerator design tools.

Feel free to reach out for contributions, issues, or feature requests!