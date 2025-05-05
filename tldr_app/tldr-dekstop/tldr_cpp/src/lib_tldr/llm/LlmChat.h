//
// Modified version of llama.cpp/examples/simple.cpp/.h
//

#ifndef LLM_CHAT_H
#define LLM_CHAT_H

#include "llama.h"
#include "ggml-backend.h"
#include "common.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct llm_result {
    bool error;
    std::string error_message;
    std::string chat_response;
};

class LlmChat {
public:
    LlmChat(std::string model_path, int n_gpu_layers = 99);
    ~LlmChat();
    bool initialize_model();
    llm_result chat_with_llm(std::string prompt, int n_predict=128);

private:
    std::string model_path;
    int n_gpu_layers;
    llama_context *ctx;
    llama_model *model;
    const llama_vocab *vocab;
    common_params params;
};

#endif //LLM_CHAT_H
