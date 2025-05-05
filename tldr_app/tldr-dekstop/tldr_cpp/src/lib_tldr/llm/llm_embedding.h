//
// Modified version of llama.cpp/examples/embedding.cpp/.h
//
#ifndef LLM_EMBEDDING_H
#define LLM_EMBEDDING_H
#include <vector>
#include "llama.h"
#include "common.h"

class llm_embedding {
public:
    llm_embedding(std::string chat_model_path);
    std::vector<float> llm_get_embeddings(std::vector<std::string_view> input_batch);
    ~llm_embedding();
private:
    llama_context * ctx;
    llama_model * model;
    const llama_vocab * vocab;
    common_params params;
};



#endif //LLM_EMBEDDING_H
