//
// Modified version of llama.cpp/examples/simple.cpp/.h
//

#ifndef LLM_CHAT_H
#define LLM_CHAT_H

#include "llama.h"
#include "common.h"
#include "LlmContextPool.h"
#include <string>
#include <vector>
#include <memory>

struct llm_result {
    bool error;
    std::string error_message;
    std::string chat_response;
};

class LlmChat {
public:
    LlmChat();
    void llm_chat_cleanup();
    bool initialize_model(const std::string& model_path);
    llm_result chat_with_llm(std::string prompt);

private:
    std::string model_path;
    int n_gpu_layers;
    llama_model *model;
    const llama_vocab *vocab;
    common_params params;
    std::vector<double> call_times_ms;
    std::vector<size_t> prompt_sizes;
    
    // Context pool for reusing contexts
    std::unique_ptr<tldr::LlmContextPool> context_pool;
};

#endif //LLM_CHAT_H
