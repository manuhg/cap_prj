# Notes

# 2024-22-24
TODOs
- Run the llama 2 7B model that has been converted to swift.
- Start quantizing models
- Try to run the model in direct PyTorch vs Swift and compare
- Try to port Phi2 to Swift

- Go through all these content
  - [x] https://www.youtube.com/watch?v=ffz-AgHZ_PQ - case for a kv cache
  - [x] https://www.youtube.com/watch?v=hMs8VNRy5Ys - decoder only inference, continuous batching speculative decoding (medusa, n-gram)
  - [x] https://www.youtube.com/watch?v=z2M8gKGYws4 - Understanding the LLM Inference Workload - Mark Moyou, NVIDIA
    - int4/FP8 quantization using AWQ has high perf improvement for smaller batch sizes (<4) with low impact of quality. But taken O(10min) for callibration
    - mmoyou@nvidia.com
  - [] https://www.youtube.com/watch?v=Mn_9W1nCFLo - LLaMa paper explained
  - [] https://www.youtube.com/watch?v=5ZlavKF_98U - faster llm serving with paged attention
  - [] https://www.youtube.com/watch?v=CVHsH_J65ok - Tailoring Small Language Models for Enterprise Use Cases
  - [] https://www.youtube.com/watch?v=af3D5WS0SGc - LLMs on Macbook Air
  - [] https://www.youtube.com/watch?v=3GEz_0ddFIo - LLMs in production
  - [] https://www.youtube.com/watch?v=2TT384U4vQg&list=WL&index=24 - Better attention layers
  - [] https://www.youtube.com/watch?v=qBFENFjKE-M&list=WL&index=5&t=9s - Accelerating LLM Inference with vLLM
  - [] https://www.youtube.com/watch?v=mYRqvB1_gRk&list=WL&index=4&t=12s - Exploring the Latency/Throughput & Cost Space for LLM Inference // Timothée Lacroix // CTO Mistral

  - [] https://arxiv.org/pdf/1811.08886 - hardware aware quantization 
  - [] https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
  - [] https://github.com/Vishesht27/KV_Cache/blob/main/model.py
  - [] https://github.com/karpathy/llm.c/blob/master/train_gpt2.py
  - [] https://news.ycombinator.com/item?id=40502090
  - [] https://github.com/huggingface/swift-transformers/blob/main/Sources/Generation/Generation.swift
  - [] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/benchmarks/index.html
  - [x] https://news.ycombinator.com/item?id=37740932: Streaming LLMs - while they can generate upto 4M tokens, since its a little bit of trick on top of sliding window attention, there is significant loss of information and might exacerbate the "lost in the middle" problem. Perhaps more useful for pure chat based use cases. But could end up repeating itself.

### For periodic memory refresh:

TTFS - time to first time (first step is slow due to no cache) = understanding the prompt + generating the first token 

Encoder only model -> generates vector embedding / summarizes the meaning but cannot generate output
Auto regressive decoding -> takes encoder’s context and all tokens generated so far and yields the next token/outout
Encoder-decoder is useful for translation, summarization, image captioning ; where the entire context needs to be obtained first
Control tokens: start, stop, code-word for the language to translate to, or complexity level/pattern, etc

This is why in decoder only models, the prompt plays the key role. It provides the much needed “context” that is lacking in a decoder only model.
Problem: no distinction between instruction and input

Activation is all about reducing memory footprint while still retaining the numerical distribution of the weights and activations obtained from the original model. So if a model is way too sparse, quantization gets harder and not easier. So its more about rescaling the range of values including the outliers.   

#### resources
100 free inference requests - https://build.nvidia.com/explore/discover 
# older notes
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
