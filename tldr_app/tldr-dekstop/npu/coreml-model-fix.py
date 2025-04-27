import coremltools as ct
import numpy as np

# Load the .mlpackage file
mlmodel = ct.models.MLModel('tldr_npu/CosineSimilarity.mlpackage')

# Example usage of the loaded model
# Assuming the model requires two inputs, similar to the original PyTorch model
example_input1 = ct.TensorType(name="input1", shape=(1, 128))
example_input2 = ct.TensorType(name="input2", shape=(1, 128))

# Since CoreML models are typically used in iOS applications, direct execution in Python is not straightforward.
# However, you can convert inputs to the required format and use the model in an iOS app or a compatible environment.

# Example data for inference
input_data1 = np.random.rand(1, 128).astype(np.float32)
input_data2 = np.random.rand(1, 128).astype(np.float32)

# Prepare inputs
inputs = {'input1': input_data1, 'input2': input_data2}

# Perform inference
output = mlmodel.predict(inputs)

# Convert NumPy arrays in the output dictionary to lists for JSON serialization
serializable_output = {k: v.tolist() for k, v in output.items()}

# Save the output
output_file_path = 'output.json'
with open(output_file_path, 'w') as f:
    import json
    json.dump(serializable_output, f)

print(f"Output saved to {output_file_path}")