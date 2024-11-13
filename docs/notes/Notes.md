# Notes
Useful repos
- vllm - https://github.com/vllm-project/vllm  
  Does not support ARM/ANE (only linux+x86 CPU, or specific NVIDIA GPU)

- ollama - https://github.com/ollama/ollama
- llamaFile - https://github.com/Mozilla-Ocho/llamafile
- llamaRag - https://github.com/Mozilla-Ocho/llamafile-rag-example
- llama.cpp - https://github.com/ggerganov/llama.cpp
- flash_attn - https://github.com/Dao-AILab/flash-attention     
    Requires nvcc
- accelerate - https://github.com/huggingface/accelerate  
   Does not support ANE
- openVINO - https://github.com/openvinotoolkit/openvino  
   OSS repo for model deployment/inference

# Research
- Apple CoreML vs pytorch + MPS  
  - coreML implementation 31x faster https://drsleep.github.io/technical/Neural-Sketching-CoreML/

# TODOS
[] Run all these models on device and try to make a benchmark of sorts
[] look at what ANE/metal api offers

[] make a device map for m1 for huggingface
[] make a flash attention implementation for m1
[] implement vllm for m1?
[] look at ollama/llama.cpp implementation of ANE/m1
[] check what more can be done


# https://huggingface.co/Salesforce/xLAM-1b-fc-r
