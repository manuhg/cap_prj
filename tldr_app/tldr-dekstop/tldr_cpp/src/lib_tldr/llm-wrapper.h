#ifndef LLM_WRAPPER_H
#define LLM_WRAPPER_H

#include <string>
#include <vector>
#include <string_view>

// Include llama.h directly for full type definitions
#include "llama.h" 
// #include "common.h" // common.h likely included by llama.h or only needed in cpp

namespace tldr {

    // Define configuration structure for LLM models
    struct LLMConfig {
        int n_gpu_layers = 100; // Number of layers to offload to GPU
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
        LlmManager();
        ~LlmManager();

        // Initialization methods
        bool initialize_chat_model(const std::string& model_path, const LLMConfig& config);
        bool initialize_embeddings_model(const std::string& model_path, const LLMConfig& config);

        // Core functionalities
        std::vector<std::vector<float>> get_embeddings(const std::vector<std::string_view>& texts);
        std::string get_chat_response(const std::string& context, const std::string& user_prompt);

        // TODO: Add getter methods if contexts/models need to be accessed externally
        // llama_context* get_chat_context() const { return chat_ctx_; }
        // llama_context* get_embeddings_context() const { return embeddings_ctx_; }

    private:
        // Disallow copy and assign
        LlmManager(const LlmManager&) = delete;
        LlmManager& operator=(const LlmManager&) = delete;

        void cleanup_chat_model();
        void cleanup_embeddings_model();

        llama_model* chat_model_ = nullptr;
        llama_context* chat_ctx_ = nullptr;
        llama_model* embeddings_model_ = nullptr;
        llama_context* embeddings_ctx_ = nullptr;
        LLMConfig chat_config_; // Store config used for chat model
        LLMConfig embeddings_config_; // Store config used for embeddings model
    };

    // Initialization function (call once)
    void initialize_llm_manager_once(const std::string& chat_model_path, const std::string& embeddings_model_path);

    // Accessor for the global manager instance (after initialization)
    LlmManager& get_llm_manager();

} // namespace tldr

#endif // LLM_WRAPPER_H
