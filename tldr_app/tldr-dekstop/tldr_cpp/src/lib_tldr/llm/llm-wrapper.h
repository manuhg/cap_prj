#ifndef LLM_WRAPPER_H
#define LLM_WRAPPER_H

#include <string>
#include <vector>
#include <string_view>


#include "LlmChat.h"
#include "LlmEmbeddings.h"

namespace tldr {

    // Define configuration structure for LLM models
    struct LLMConfig {
        int n_gpu_layers = 0; // Number of layers to offload to GPU
        int n_ctx = 2048;     // Context size
        int n_batch = 512;    // Batch size for prompt processing
        enum llama_pooling_type pooling_type = LLAMA_POOLING_TYPE_MEAN; // Use 'enum' tag, Default to MEAN pooling

        // --- Add Sampler Parameters ---
        int top_k = 40;           // Default top-k
        float top_p = 0.95f;        // Default top-p
        float temp = 0.8f;          // Default temperature
        // float repeat_penalty = 1.1f; // Example for repetition penalty (if added later)
        // --- End Sampler Parameters ---

        // Add other relevant parameters as needed
    };

    class LlmManager {
    public:
        LlmManager(const std::string &chat_model_path, const std::string &embeddings_model_path);
        ~LlmManager();

        // Initialization methods
        bool initialize_chat_model();
        bool initialize_embeddings_model();

        // Core functionalities
        std::vector<std::vector<float>> get_embeddings(const std::vector<std::string_view>& texts);
        std::string get_chat_response(const std::string& context, const std::string& user_prompt);


    private:
        LlmChat chat;
        LlmEmbeddings embedding;
    };

    // Initialization function (call once)
    void initialize_llm_manager_once();

    // Accessor for the global manager instance (after initialization)
    LlmManager& get_llm_manager();

} // namespace tldr

#endif // LLM_WRAPPER_H
