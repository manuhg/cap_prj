//
// Modified version of llama.cpp/examples/embedding.cpp/.h
//
#ifndef LLM_EMBEDDING_H
#define LLM_EMBEDDING_H
#include <vector>
#include <memory>
#include "llama.h"
#include "common.h"
#include "LlmContextPool.h"

class LlmEmbeddings {
public:
    LlmEmbeddings(std::string model_path);
    bool initialize_model();
    void embedding_cleanup();
    std::vector<std::vector<float>> llm_get_embeddings(std::vector<std::string_view> input_batch);
private:
    std::string model_path;
    llama_model * model;
    const llama_vocab * vocab;
    common_params params;
    // store total runtime in milliseconds for each embeddings call
    std::vector<double> call_times_ms;
    std::vector<size_t> batch_sizes;
    std::vector<size_t> prompt_sizes;
    
    // Context pool for reusing contexts
    std::unique_ptr<tldr::LlmContextPool> context_pool;
};



#endif //LLM_EMBEDDING_H
