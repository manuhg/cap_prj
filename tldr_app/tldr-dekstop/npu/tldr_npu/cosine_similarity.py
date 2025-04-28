import torch
import torch.nn as nn
import numpy as np
import sys

# Define the PyTorch model
class CosineSimilarityModel(nn.Module):
    def __init__(self):
        super(CosineSimilarityModel, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1)  # Compare across the feature dimension (D)

    def forward(self, input1, input2):
        # input1 shape: (1, D)
        # input2 shape: (M, D)
        # Expand input1 to match batch size M of input2
        m = input2.shape[0]  # Get batch size M
        input1_expanded = input1.expand(m, -1) # Shape becomes (M, D)
        # Calculate similarity between expanded input1 and input2
        # Output shape: (M,)
        return self.cos(input1_expanded, input2)

# --- Test the model with sample data ---
model = CosineSimilarityModel()
model.eval()

VECTOR_DIM = 128
BATCH_SIZE = 10 # Example batch size for input2

# Dummy inputs to test the model
example_input1 = torch.randn(1, VECTOR_DIM)  # Base vector (1, D)
example_input2 = torch.randn(BATCH_SIZE, VECTOR_DIM) # Batch of vectors (M, D)

# Run a simple test
with torch.no_grad():
    similarities = model(example_input1, example_input2)
    print(f"Input1 shape: {example_input1.shape}")
    print(f"Input2 shape: {example_input2.shape}")
    print(f"Output similarities shape: {similarities.shape}")
    print(f"Sample similarities: {similarities[:5].tolist()}...") # Print first 5

# Run test with identical vectors (should be 1.0)
identical_test_input1 = torch.ones(1, VECTOR_DIM)
identical_test_input2 = torch.ones(BATCH_SIZE, VECTOR_DIM)
with torch.no_grad():
    similarities = model(identical_test_input1, identical_test_input2)
    print(f"\nSimilarity between identical vectors (all should be 1.0): {similarities.tolist()}")

# Run test with orthogonal vectors (first vector in batch should be 0.0)
orthogonal_input1 = torch.zeros(1, VECTOR_DIM)
orthogonal_input1[0, 0] = 1.0
orthogonal_input2 = torch.zeros(BATCH_SIZE, VECTOR_DIM)
orthogonal_input2[0, 1] = 1.0 # Make first vector orthogonal
orthogonal_input2[1:, 0] = 1.0 # Make others identical to input1
with torch.no_grad():
    similarities = model(orthogonal_input1, orthogonal_input2)
    print(f"\nSimilarity with mixed batch (first should be 0.0, others 1.0): {similarities.tolist()}")

# --- Attempt to convert to CoreML ---
try:
    import coremltools as ct
    
    # Trace the model with example inputs
    traced_model = torch.jit.trace(model, (example_input1, example_input2))
    
    # Define input shapes for CoreML conversion
    # For '.mlmodel', we might need fixed shapes or specific enumerations
    input1_shape_fixed = ct.TensorType(name="input1", shape=example_input1.shape) # Shape (1, 128)
    input2_shape_fixed = ct.TensorType(name="input2", shape=example_input2.shape) # Shape (10, 128) for this example
    
    print(f"\nConverting to .mlmodel (neuralnetwork) with fixed input shapes:")
    print(f"  input1: {example_input1.shape}")
    print(f"  input2: {example_input2.shape}")

    # Convert to Core ML (.mlmodel format)
    mlmodel = ct.convert(
        traced_model,
        # Use fixed shapes derived from example inputs for neuralnetwork format
        inputs=[input1_shape_fixed, input2_shape_fixed],
        # outputs=[ct.TensorType(name="similarities")], # Optional: name output
        convert_to="neuralnetwork",  # Use older, potentially more stable format
        # compute_units=ct.ComputeUnit.ALL, # Not applicable for neuralnetwork format
    )
    
    # Save the model
    output_filename = "CosineSimilarityBatched.mlmodel" # Change extension
    mlmodel.save(output_filename)
    print(f"CoreML model saved as {output_filename}")

    # Optional: Inspect the converted model spec
    spec = mlmodel.get_spec()
    print("\nCoreML Model Inputs:")
    for input_desc in spec.description.input:
        print(f"  Name: {input_desc.name}, Type: {input_desc.type}")
    print("CoreML Model Outputs:")
    for output_desc in spec.description.output:
        print(f"  Name: {output_desc.name}, Type: {output_desc.type}")

except ImportError:
    print("\ncoremltools not found. Skipping CoreML conversion.")
except Exception as e:
    print(f"\nError converting to CoreML: {e}")