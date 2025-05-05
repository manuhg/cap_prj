#ifndef LLM_WRAPPER_H
#define LLM_WRAPPER_H

#include "llama.h" // Required for llama types
#include <string> // Required for model paths potentially
#include <vector> // Required for get_embeddings return type

namespace tldr {

    class LlmManager {
    public:
        LlmManager();
        ~LlmManager();

        // Initialization methods
        bool initialize_chat_model(const std::string& model_path);
        bool initialize_embeddings_model(const std::string& model_path);

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
    };

} // namespace tldr

#endif // LLM_WRAPPER_H
