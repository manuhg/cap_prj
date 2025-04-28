import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

# Constants
VECTOR_DIM = 128
BATCH_SIZE = 10 # Example batch size for testing

# Define the PyTorch model for batched cosine similarity
class CosineSimilarityModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Epsilon for numerical stability is handled by F.cosine_similarity

    def forward(self, input1, input2):
        # input1: (1, D) # Base vector
        # input2: (M, D) # Batch of comparison vectors
        
        # Use F.cosine_similarity. It handles broadcasting between (1, D) and (M, D)
        # when dim=1 is specified.
        # It computes the similarity between input1 and each row of input2.
        similarity = F.cosine_similarity(input1, input2, dim=1) # Output shape (M,)
        
        return similarity

# --- Create dummy data and test the PyTorch model ---

# Example usage:
model = CosineSimilarityModel()
model.eval()

# Example input tensors (batch size = BATCH_SIZE)
example_input1 = torch.randn(1, VECTOR_DIM)
example_input2 = torch.randn(BATCH_SIZE, VECTOR_DIM)

print(f"Input1 shape: {example_input1.shape}")
print(f"Input2 shape: {example_input2.shape}")

# Run inference
with torch.no_grad():
    output_similarities = model(example_input1, example_input2)

print(f"Output similarities shape: {output_similarities.shape}")
print(f"Sample similarities: {output_similarities.numpy()[:5]}...")

# Test cases
# 1. Identical vectors (similarity should be 1.0)
vec1 = torch.ones(1, VECTOR_DIM)
vec2_batch = torch.ones(BATCH_SIZE, VECTOR_DIM)
with torch.no_grad():
    sim_identical = model(vec1, vec2_batch)
print(f"\nSimilarity between identical vectors (all should be 1.0): {sim_identical.numpy()}")

# 2. Orthogonal vector (similarity should be 0.0)
# Create an orthogonal vector for the first element in the batch
vec1_ortho = torch.zeros(1, VECTOR_DIM)
vec1_ortho[0, 0] = 1.0 # Example: [1, 0, 0, ...]
vec2_batch_mixed = torch.ones(BATCH_SIZE, VECTOR_DIM) # Base is all ones
vec2_batch_mixed[0, :] = 0.0 # First comparison vector is all zeros
vec2_batch_mixed[0, 1] = 1.0 # Make it orthogonal to [1, 0, 0, ...], e.g. [0, 1, 0, ...]
with torch.no_grad():
    sim_mixed = model(vec1_ortho, vec2_batch_mixed)
print(f"\nSimilarity with mixed batch (first should be 0.0, others near 0): {sim_mixed.numpy()}")

# --- Attempt to convert to CoreML ---
try:
    import coremltools as ct
    
    # Trace the model with example inputs
    traced_model = torch.jit.trace(model, (example_input1, example_input2))
    
    # Define input shapes for CoreML conversion
    # Use RangeDim for flexible batch size M (1 to 1024)
    input1_shape = ct.TensorType(name="input1", shape=(1, VECTOR_DIM))
    input2_shape = ct.TensorType(name="input2", shape=(ct.RangeDim(1, 1024), VECTOR_DIM)) # Set max batch size
    
    print(f"\nConverting to .mlpackage (mlprogram) with flexible input shapes:")
    print(f"  input1: {example_input1.shape}")
    print(f"  input2: {example_input2.shape}")

    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        inputs=[input1_shape, input2_shape],
        # outputs=[ct.TensorType(name="similarities")], # Optional: name output
        convert_to="mlprogram",  # Use ML Program for NPU
        compute_units=ct.ComputeUnit.ALL,
    )
    
    # Save the model
    output_filename = "CosineSimilarityBatched.mlpackage"
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