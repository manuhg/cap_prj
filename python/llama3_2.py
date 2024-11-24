import torch
from transformers import pipeline

custom_model_path = "../model_weights/llm-weights/Llama-3.2-1B"

pipe = pipeline(
    "text-generation",
    model=custom_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

pipe("The key to life is")
