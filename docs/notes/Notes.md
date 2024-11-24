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


# Ingredients of RAG



- https://python.langchain.com/docs/tutorials/rag/, https://www.pingcap.com/article/building-a-rag-application-from-scratch-a-beginners-guide/
  - Load the document, chunk them and store them using DocumentLoader
  - Embed the chunk and store it along with the indexes and metadata (like beginning, middle, end of doc, etc) in VectorStore (Sannu was using an additional similarity score with known keywords to enhance the similarity search results) - VectorStore
    - Inverted indexing: map text to embed location for easier text oriented search 
    - Retrieve relevant splits for a user query using a Retriever (vector similarity search)
  - Evaluate the quality(relevance) of results using an Evaluator (ex; https://github.com/explodinggradients/ragas, https://docs.confident-ai.com/docs/guides-rag-evaluation) (ref doc on eval : https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)




Vector Search
- Use faiss
- Use Redis or PostgreSQL (with pgvector enabled)
- Use Postgres+pgai - https://github.com/timescale/pgai, https://www.timescale.com/blog/vector-databases-are-the-wrong-abstraction/
- Try to see if faiss can be configured to use Redis/Postgres
- Try to see if Redis can store its keys & values in gpu memory (for m1, due to shared memory, memcpy could be avoided)
- Use NLEmbedding from CoreML (https://developer.apple.com/documentation/naturallanguage/finding-similarities-between-pieces-of-text?language=objc)
-


Coreml reference - https://developer.apple.com/documentation/coreml/mltensor/matmul(_:)
https://github.com/hollance/neural-engine/blob/master/docs/ane-vs-gpu.md
Model conversion - https://github.com/pytorch/torchtune?tab=readme-ov-file
https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html


Info on Vector search: https://weaviate.io/blog/vector-search-explained

Quantized models
https://huggingface.co/TheBloke/phi-2-GGUF
https://huggingface.co/QuantFactory/Llama-3.2-1B-Instruct-GGUF
https://huggingface.co/lmstudio-community/Llama-3.2-1B-Instruct-GGUF
https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/tree/main
https://huggingface.co/bartowski/Phi-3.5-mini-instruct_Uncensored-GGUF/tree/main


Reads on quantization: https://symbl.ai/developers/blog/a-guide-to-quantization-in-llms/
https://www.tensorops.ai/post/what-are-quantized-llms
https://www.datacamp.com/tutorial/quantization-for-large-language-models
https://huggingface.co/posts/macadeliccc/247190826659941


Model conversion:
While model conversion makes sense for traditional PyTorch/Tensorflow packages, in case of LLMs, this leads to inefficiency and overhead. Capabilities like kv-caching is not efficient since it’s done using pythonic code.
Hence, in this case, it is better to use optimized implementation of each model, as defined in projects like llama.cpp



Way ahead: use the GGUF format, with quantization between q2k to q5.



Chunking:
Start with fixed size chunking of 256 tokens or one paragraph (whichever is lower) with some overlap
https://www.pinecone.io/learn/chunking-strategies/





Discussion on ANE support for GGML - https://github.com/ggerganov/llama.cpp/discussions/336
https://github.com/apple/ml-ane-transformers/tree/main
Exposing Swift APIs to C++
https://www.swift.org/documentation/cxx-interop/#exposing-swift-apis-to-c
Apple metal is similar to CUDA and uses the GPU but not the ANE. To use ANE will have to use coreML and write code in Swift

Next steps:
Make a cli that
- Reads text files (convert pdf to text?)
- Embed them into tokens ( llama.create_embedding)
- Store them in psql
- Do vector search over text
- Feed to phi 2
- Get some results

Use faiss. Try to port the cuda code to MPS (https://github.com/MEHDI342/CUDAM and even potentially ANE)


- Identify scopes of improvement
  - llama.cpp optimizations
  - Vector search optimizations (leveraging ANE for vector search)
