# Notes

# 2024-22-24
TODOs
- Run the llama 2 7B model that has been converted to swift.
- Start quantizing models
- Try to run the model in direct PyTorch vs Swift and compare
- Try to port Phi2 to Swift

- Go through all these content
  - https://www.youtube.com/watch?v=ffz-AgHZ_PQ&t=17s
  - https://www.youtube.com/watch?v=hMs8VNRy5Ys&t=545s
  - https://www.youtube.com/watch?v=Mn_9W1nCFLo
  - https://www.youtube.com/watch?v=5ZlavKF_98U
  - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/benchmarks/index.html
  - https://github.com/Vishesht27/KV_Cache/blob/main/model.py
  - https://www.youtube.com/watch?v=CVHsH_J65ok
  - https://github.com/karpathy/llm.c/blob/master/train_gpt2.py
  - https://news.ycombinator.com/item?id=40502090
  - https://github.com/huggingface/swift-transformers/blob/main/Sources/Generation/Generation.swift

### For periodic memory refresh:

TTFS - time to first time (first step is slow due to no cache)

Encoder only model -> generates vector embedding / summarizes the meaning but cannot generate output
Auto regressive decoding -> takes encoder’s context and all tokens generated so far and yields the next token/outout
Encoder-decoder is useful for translation, summarization, image captioning ; where the entire context needs to be obtained first
Control tokens: start, stop, code-word for the language to translate to, or complexity level/pattern, etc

This is why in decoder only models, the prompt plays the key role. It provides the much needed “context” that is lacking in a decoder only model.
Problem: no distinction between instruction and input


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
