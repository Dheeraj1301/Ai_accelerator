# Save as generate_data.py and run once
import numpy as np

# Generate a random array of 100 integers between 0 and 255
data = np.random.randint(0, 255, 100)

# Save it to the data directory as input_sample.npy
np.save("data/input_sample.npy", data)
print("✅ input_sample.npy generated and saved in 'data/' directory.")
