//
// Modified version of llama.cpp/examples/embedding.cpp/.h
//
#ifndef LLM_EMBEDDING_H
#define LLM_EMBEDDING_H
#include <vector>
#include "llama.h"
#include "common.h"

class LlmEmbeddings {
public:
    LlmEmbeddings(std::string model_path);
    bool initialize_model();
    void embedding_cleanup();
    std::vector<std::vector<float>> llm_get_embeddings(std::vector<std::string> input_batch);
private:
    std::string model_path;
    llama_context * ctx;
    llama_model * model;
    const llama_vocab * vocab;
    common_params params;
};



#endif //LLM_EMBEDDING_H
