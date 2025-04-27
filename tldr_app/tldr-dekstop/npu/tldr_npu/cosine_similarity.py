import torch
import torch.nn as nn
import numpy as np
import sys

# Define the PyTorch model
class CosineSimilarityModel(nn.Module):
    def __init__(self):
        super(CosineSimilarityModel, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1)  # Compare across the feature dimension

    def forward(self, input1, input2):
        return self.cos(input1, input2)

# Test the model with sample data
model = CosineSimilarityModel()
model.eval()

# Dummy inputs to test the model
example_input1 = torch.randn(1, 128)  # 128-dimensional vector
example_input2 = torch.randn(1, 128)

# Run a simple test
with torch.no_grad():
    similarity = model(example_input1, example_input2)
    print(f"Similarity between two random vectors: {similarity.item()}")

# Run test with identical vectors (should be 1.0)
identical_test = torch.ones(1, 128)
with torch.no_grad():
    similarity = model(identical_test, identical_test)
    print(f"Similarity between identical vectors: {similarity.item()}")

# Run test with orthogonal vectors (should be 0.0)
orthogonal1 = torch.zeros(1, 128)
orthogonal1[0, 0] = 1.0
orthogonal2 = torch.zeros(1, 128)
orthogonal2[0, 1] = 1.0
with torch.no_grad():
    similarity = model(orthogonal1, orthogonal2)
    print(f"Similarity between orthogonal vectors: {similarity.item()}")


try:
    import coremltools as ct

    # Trace the model
    traced_model = torch.jit.trace(model, (example_input1, example_input2))

    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input1", shape=example_input1.shape),
            ct.TensorType(name="input2", shape=example_input2.shape),
        ],
        convert_to="mlprogram",  # Use ML Program to allow running on Neural Engine
        compute_units=ct.ComputeUnit.ALL,
    )

    # Save the model
    mlmodel.save("CosineSimilarity.mlmodel")
    print("CoreML model saved as CosineSimilarity.mlmodel")
except Exception as e:
    print(f"Error converting to CoreML: {e}")
    print("CoreML conversion skipped, but PyTorch model works correctly.")