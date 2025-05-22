#include <iostream>
#include <string>
#include <vector>
#include <mutex>

#include "llm-wrapper.h"

#include "../constants.h"

namespace tldr {
    // --- Static Instance and Initialization ---

    // Static instance, managed internally
    static LlmManager g_llm_manager_instance = LlmManager(CHAT_MODEL_PATH, EMBEDDINGS_MODEL_PATH);
    static std::once_flag g_init_flag;

    // Accessor implementation
    LlmManager &get_llm_manager() {
        // Ensure initialization has happened (optional, depends on usage pattern)
        // std::call_once(g_init_flag, ...); // Could re-trigger init logic if needed
        return g_llm_manager_instance;
    }

    // Moved initialization function
    void initialize_llm_manager_once() {
        std::call_once(g_init_flag, [&]() {
            // llama_backend_init();

            std::cout << "Initializing chat model..." << std::endl;
            if (!g_llm_manager_instance.initialize_chat_model()) {
            std::cerr << "Failed to initialize chat model." << std::endl;
            // Handle initialization failure
            }

            std::cout << "Initializing embeddings model..." << std::endl;
            if (!g_llm_manager_instance.initialize_embeddings_model()) {
                std::cerr << "Failed to initialize embeddings model." << std::endl;
                // Handle initialization failure
            }
        });
    }

    // --- LlmManager Class Implementation ---

    LlmManager::LlmManager(const std::string &chat_model_path,
                           const std::string &embeddings_model_path): chat(chat_model_path),
                                                                      embedding(embeddings_model_path) {}

    bool LlmManager::initialize_chat_model() {
        try {
            chat.initialize_model();
            return true; // Return true on success
        } catch (const std::exception &e) {
            std::cerr << "Error: Failed to load embedding model.:" << e.what() << std::endl;
            return false;
        }
    }

    bool LlmManager::initialize_embeddings_model() {
        try {
            embedding.initialize_model();
            return true; // Return true on success
        } catch (const std::exception &e) {
            std::cerr << "Error: Failed to load embedding model.:" << e.what() << std::endl;
            return false;
        }
    }

    std::vector<std::vector<float>> LlmManager::get_embeddings(const std::vector<std::string_view> &texts) {
        return embedding.llm_get_embeddings(texts);
    }

    std::string LlmManager::get_chat_response(const std::string &context, const std::string &prompt) {
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
