# tests/test_pipeline_latency.py
import time
import numpy as np
from python_sim.simulate_pipeline import simulate_pipeline

def test_latency():
    x = np.random.randint(0, 255, 1000)
    start = time.time()
    simulate_pipeline(x)
    assert time.time() - start < 1.0
