#include <iostream>
#include <string>
#include <vector>
#include <mutex>

#include "llm-wrapper.h"

#include "../constants.h"

namespace tldr {
    // --- Static Instance and Initialization ---

    // Static instance, managed internally
    static LlmManager g_llm_manager_instance = LlmManager();
    static std::once_flag g_init_flag;

    // Accessor implementation
    LlmManager &get_llm_manager() {
        // Ensure initialization has happened (optional, depends on usage pattern)
        // std::call_once(g_init_flag, ...); // Could re-trigger init logic if needed
        return g_llm_manager_instance;
    }

    // Moved initialization function
    void initialize_llm_manager_once(const std::string& chat_model_path,
                                   const std::string& embeddings_model_path) {
        std::call_once(g_init_flag, [&]() {
            if (chat_model_path.empty()) {
                throw std::invalid_argument("Chat model path cannot be empty");
            }
            if (embeddings_model_path.empty()) {
                throw std::invalid_argument("Embeddings model path cannot be empty");
            }

            // Recreate the manager instance with the new paths
            g_llm_manager_instance = LlmManager();
            if (chat_model_path.empty()) {
                throw std::invalid_argument("Chat model path cannot be empty");
            }
            if (embeddings_model_path.empty()) {
                throw std::invalid_argument("Embeddings model path cannot be empty");
            }

            // Initialize models with their respective paths
            if (!g_llm_manager_instance.initialize_chat_model(chat_model_path)) {
                throw std::runtime_error("Failed to initialize chat model");
            }
            if (!g_llm_manager_instance.initialize_embeddings_model(embeddings_model_path)) {
                throw std::runtime_error("Failed to initialize embeddings model");
            }


            std::cout << "LLM Manager initialized with models:" << std::endl;
            std::cout << "- Chat model: " << chat_model_path << std::endl;
            std::cout << "- Embeddings model: " << embeddings_model_path << std::endl;
        });
    }

    // --- LlmManager Class Implementation ---

    LlmManager::LlmManager() {
        // do nothing
    }

    bool LlmManager::initialize_chat_model(const std::string& model_path) {
    try {
        return chat.initialize_model(model_path);
    } catch (const std::exception &e) {
        std::cerr << "Error: Failed to load chat model: " << e.what() << std::endl;
        return false;
    }
}

    bool LlmManager::initialize_embeddings_model(const std::string& model_path) {
    try {
        return embedding.initialize_model(model_path);
    } catch (const std::exception &e) {
        std::cerr << "Error: Failed to load embeddings model: " << e.what() << std::endl;
        return false;
    }
}

    std::vector<std::vector<float>> LlmManager::get_embeddings(const std::vector<std::string_view> &texts) {
        return embedding.llm_get_embeddings(texts);
    }

    std::string LlmManager::get_chat_response(const std::string &context, const std::string &prompt) {
        std::cout<<"\nGenerating Chat LLM Response ..."<<std::endl;
        const std::string system_prompt =
                "You are a helpful AI Assistant. Go through the given context and answer the user's questions. Keep the answers short and precise.";
        std::string formatted_prompt = "<|system|>\n" + system_prompt + "\n<|context|>\n" + context + "\n<|user|>\n" +
                                       prompt + "\n<|assistant|>\n";

        // A simpler alternative if the model doesn't use special tokens:
        // std::string formatted_prompt = system_prompt + "\n\nUser: " + user_prompt + "\nAssistant: ";


        auto result = chat.chat_with_llm(formatted_prompt);
        if (result.error) {
            std::cerr << "Error: " << result.error << std::endl;
            return "Error obtaining result from the LLM!";
        }
        return result.chat_response;
    }

    void LlmManager::cleanup() {
        chat.llm_chat_cleanup();
        embedding.embedding_cleanup();
    }

} // namespace tldr
