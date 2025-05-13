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

    /**
     * LlmManager - Manages LLM models and context pools for both chat and embeddings
     * 
     * This class handles the initialization and management of LLM models and their
     * associated context pools. It provides a high-level interface for getting
     * embeddings and chat responses, while handling the low-level details of
     * context reuse and management.
     */
    class LlmManager {
    public:
        /**
         * Constructor - initializes the chat and embedding models
         * @param chat_model_path Path to the chat model file
         * @param embeddings_model_path Path to the embeddings model file
         */
        LlmManager(const std::string &chat_model_path, const std::string &embeddings_model_path);

        // Initialization methods
        /**
         * Initialize the chat model and its context pool
         * @return true if successful, false otherwise
         */
        bool initialize_chat_model();
        
        /**
         * Initialize the embeddings model and its context pool
         * @return true if successful, false otherwise
         */
        bool initialize_embeddings_model();

        // Core functionalities
        /**
         * Get embeddings for a batch of texts
         * @param texts The texts to embed
         * @return A vector of embedding vectors
         */
        std::vector<std::vector<float>> get_embeddings(const std::vector<std::string_view>& texts);
        
        /**
         * Get a chat response for a given context and user prompt
         * @param context The context to use for the chat
         * @param user_prompt The user's prompt
         * @return The generated response
         */
        std::string get_chat_response(const std::string& context, const std::string& user_prompt);

        /**
         * Clean up all resources
         * This will free all contexts in the pools and then free the models
         */
        void cleanup();

    private:
        LlmChat chat;         // Chat model and its context pool
        LlmEmbeddings embedding; // Embeddings model and its context pool
    };

    // Initialization function (call once)
    void initialize_llm_manager_once();

    // Accessor for the global manager instance (after initialization)
    LlmManager& get_llm_manager();

} // namespace tldr

#endif // LLM_WRAPPER_H
