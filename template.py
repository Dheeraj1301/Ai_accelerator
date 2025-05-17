import os

# Define your project name
project_name = "modular_ai_accelerator"

# Define all file paths to be created
list_of_files = [
    f"{project_name}/docs/architecture_diagram.png",  # Can be empty or placeholder

    f"{project_name}/chiplets/conv_chiplet.v",
    f"{project_name}/chiplets/matmul_chiplet.v",
    f"{project_name}/chiplets/activation_chiplet.v",

    f"{project_name}/hdl/top_module.v",

    f"{project_name}/python_sim/simulate_pipeline.py",
    f"{project_name}/python_sim/chiplet_classes.py",
    f"{project_name}/python_sim/visualize.py",

    f"{project_name}/models/cnn_model.py",

    f"{project_name}/tests/test_chiplets.py",
    f"{project_name}/tests/test_pipeline_latency.py",

    f"{project_name}/data/input_sample.npy",

    f"{project_name}/requirements.txt",
    f"{project_name}/README.md",
]

# Loop through each file path
for filepath in list_of_files:
    filepath = os.path.normpath(filepath)
    filedir, filename = os.path.split(filepath)

    # Create the directory if it doesn't exist
    if filedir and not os.path.exists(filedir):
        os.makedirs(filedir, exist_ok=True)
        print(f"✅ Created directory: {filedir}")

    # Create the file if it doesn't exist or is empty
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "wb" if filename.endswith(".npy") else "w") as f:
            pass  # Creates an empty file (binary if .npy)
        print(f"📄 Created empty file: {filepath}")
    else:
        print(f"⚠️ File already exists and is not empty: {filepath}")
