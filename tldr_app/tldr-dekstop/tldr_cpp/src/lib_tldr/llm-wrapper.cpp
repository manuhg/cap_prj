#include "llm-wrapper.h"
#include "constants.h"
#include "llama.h"

#include <iostream>
#include <string>
#include <stdexcept> // For exceptions
#include <vector>
#include <thread> // For thread count
#include <algorithm> // For std::min

namespace tldr {

// LlmManager Implementation

LlmManager::LlmManager() : chat_model_(nullptr), chat_ctx_(nullptr), embeddings_model_(nullptr), embeddings_ctx_(nullptr) {
    // Constructor: Initialize pointers to null
}

LlmManager::~LlmManager() {
    // Destructor: Ensure cleanup is called
    cleanup_chat_model();
    cleanup_embeddings_model();
}

bool LlmManager::initialize_chat_model(const std::string& model_path) {
    // Cleanup existing model first if any
    cleanup_chat_model();

    llama_model_params model_params = llama_model_default_params();
    // Customize model params if needed, e.g., model_params.n_gpu_layers = ...

    chat_model_ = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!chat_model_) {
        std::cerr << "Failed to load chat model from " << model_path << std::endl;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    // Customize context params if needed, e.g., ctx_params.n_ctx = 2048;
    ctx_params.seed = -1; // Use random seed

    chat_ctx_ = llama_new_context_with_model(chat_model_, ctx_params);
    if (!chat_ctx_) {
        std::cerr << "Failed to create chat context" << std::endl;
        llama_free_model(chat_model_);
        chat_model_ = nullptr;
        return false;
    }

    std::cout << "Chat model initialized successfully." << std::endl;
    return true;
}

bool LlmManager::initialize_embeddings_model(const std::string& model_path) {
    // Cleanup existing model first if any
    cleanup_embeddings_model();

    llama_model_params model_params = llama_model_default_params();
    // Customize model params if needed
    model_params.embedding = true; // Ensure model is loaded for embeddings

    embeddings_model_ = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!embeddings_model_) {
        std::cerr << "Failed to load embeddings model from " << model_path << std::endl;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    // Customize context params
    ctx_params.seed = -1;
    ctx_params.logits_all = true; // Needed for embeddings extraction? Maybe not strictly required just for llama_get_embeddings
    ctx_params.embedding = true; // Context is for embeddings

    embeddings_ctx_ = llama_new_context_with_model(embeddings_model_, ctx_params);
    if (!embeddings_ctx_) {
        std::cerr << "Failed to create embeddings context" << std::endl;
        llama_free_model(embeddings_model_);
        embeddings_model_ = nullptr;
        return false;
    }

    std::cout << "Embeddings model initialized successfully." << std::endl;
    return true;
}

void LlmManager::cleanup_chat_model() {
    if (chat_ctx_) {
        llama_free(chat_ctx_);
        chat_ctx_ = nullptr;
    }
    if (chat_model_) {
        llama_free_model(chat_model_);
        chat_model_ = nullptr;
    }
}

void LlmManager::cleanup_embeddings_model() {
    if (embeddings_ctx_) {
        llama_free(embeddings_ctx_);
        embeddings_ctx_ = nullptr;
    }
    if (embeddings_model_) {
        llama_free_model(embeddings_model_);
        embeddings_model_ = nullptr;
    }
}

// --- Embedding Generation --- 
std::vector<std::vector<float>> LlmManager::get_embeddings(const std::vector<std::string_view>& texts) {
    if (!embeddings_model_ || !embeddings_ctx_) {
        throw std::runtime_error("Embeddings model or context not initialized in LlmManager::get_embeddings.");
    }
    // Ensure embeddings model type (redundant if loaded with embedding=true, but good check)
    if (!llama_model_has_embeddings(embeddings_model_)) {
         throw std::runtime_error("Model loaded in embeddings_model_ does not support embeddings.");
    }

    std::vector<std::vector<float>> all_embeddings;
    all_embeddings.reserve(texts.size());
    const int n_embed = llama_n_embd(llama_get_model(embeddings_ctx_));
    const int n_ctx = llama_n_ctx(embeddings_ctx_);
    const int n_threads = std::max(1, (int)std::thread::hardware_concurrency()); // Ensure at least 1 thread

    // Temporary buffer for tokens
    std::vector<llama_token> tokens;

    for (const auto& text : texts) {
        if (text.empty()) {
            std::cerr << "Warning: Skipping empty string for embedding generation." << std::endl;
            continue;
        }

        // 1. Tokenize (add BOS based on model specifics - often true for embeddings)
        // Reserve space generously
        tokens.resize(text.size() + 1); 
        int n_tokens = llama_tokenize(embeddings_model_, text.c_str(), text.length(), tokens.data(), tokens.size(), true, false);
        if (n_tokens < 0) {
             std::cerr << "Error tokenizing text for embeddings: '" << text.substr(0, 100) << "...'" << std::endl;
             continue; // Skip this text
        }
        tokens.resize(n_tokens);

        if (n_tokens == 0) {
            std::cerr << "Warning: Text resulted in zero tokens: '" << text.substr(0, 100) << "...'" << std::endl;
            continue;
        }
        
        if (n_tokens > n_ctx) {
            std::cerr << "Error: Token count (" << n_tokens << ") exceeds context size (" << n_ctx << ") for text: '" << text.substr(0, 50) << "...' Truncating." << std::endl;
            n_tokens = n_ctx; // Truncate tokens to fit context
            tokens.resize(n_tokens);
        }

        // 2. Clear KV Cache (critical for independent text embeddings)
        llama_kv_cache_clear(embeddings_ctx_);

        // 3. Evaluate the tokens
        if (llama_eval(embeddings_ctx_, tokens.data(), n_tokens, 0, n_threads)) {
           std::cerr << "Error evaluating tokens for text embeddings: '" << text.substr(0, 100) << "...'" << std::endl;
           continue; // Skip this text
       }

        // 4. Extract the embeddings
        const float *embedding_ptr = llama_get_embeddings(embeddings_ctx_);
        if (!embedding_ptr) {
             std::cerr << "Error getting embeddings pointer after evaluation for text: '" << text.substr(0, 100) << "...'" << std::endl;
             continue; // Skip this text
        }
        
        // Copy the embedding vector
        all_embeddings.emplace_back(embedding_ptr, embedding_ptr + n_embed);
    }

    return all_embeddings;
}

// --- Chat Generation ---
std::string LlmManager::get_chat_response(const std::string& context, const std::string& user_prompt) {
    if (!chat_model_ || !chat_ctx_) {
        throw std::runtime_error("Chat model or context not initialized in LlmManager::get_chat_response.");
    }

    // --- System Prompt & Formatting (Highly Model Dependent) ---
    // Incorporate the provided context
    const std::string system_prompt = "You are a helpful AI Assistant. Use the following context to answer the user's question.";
    // Example formatting (adjust based on model fine-tuning):
    std::string formatted_prompt = "<|system|>\n" + system_prompt + "\n\nContext:\n" + context + "\n\nUser Question:\n" + user_prompt + "\n<|assistant|>\n";
    // Simpler alternative:
    // std::string formatted_prompt = system_prompt + "\n\nContext:\n" + context + "\n\nUser: " + user_prompt + "\nAssistant: ";

    // 1. Tokenize the formatted prompt (add BOS? Depends on model. Assume true)
    // Special tokens like <|system|> might need special handling if not part of vocab
    std::vector<llama_token> prompt_tokens;
    prompt_tokens.resize(formatted_prompt.size() + 1); // Max possible size
    int n_prompt_tokens = llama_tokenize(chat_model_, formatted_prompt.c_str(), formatted_prompt.length(), prompt_tokens.data(), prompt_tokens.size(), true, false);
     if (n_prompt_tokens < 0) {
        std::cerr << "Error: Prompt tokenization failed. Result: " << n_prompt_tokens << std::endl;
        // Consider logging the prompt here for debugging
        return "[Error: Prompt tokenization failed]";
    }
    prompt_tokens.resize(n_prompt_tokens);

    if (n_prompt_tokens == 0) {
         std::cerr << "Error: Prompt resulted in zero tokens." << std::endl;
         return "[Error: Prompt resulted in zero tokens]";
    }

    // Check prompt length against context size
    const int n_ctx = llama_n_ctx(chat_ctx_);
    if (n_prompt_tokens >= n_ctx) {
         std::cerr << "Error: Prompt token count (" << n_prompt_tokens << ") exceeds or equals context size (" << n_ctx << ")." << std::endl;
         return "[Error: Prompt too long for context window]";
    }

    // --- Generation Setup ---
    const int max_gen_tokens = std::min(512, n_ctx - n_prompt_tokens); // Max new tokens, ensure fits context
    const int n_threads = std::max(1, (int)std::thread::hardware_concurrency());
    const llama_token eos_token = llama_token_eos(llama_get_model(chat_ctx_)); // End-of-sequence token

    std::string response_text = "";
    int n_past = 0; // Number of tokens evaluated so far
    int n_remain = max_gen_tokens;

    // --- IMPORTANT: Clear KV Cache before starting generation for a new independent prompt ---
    llama_kv_cache_clear(chat_ctx_);

    // 2. Evaluate the prompt tokens first
    if (llama_eval(chat_ctx_, prompt_tokens.data(), n_prompt_tokens, n_past, n_threads)) {
       std::cerr << "Error evaluating prompt tokens." << std::endl;
       return "[Error: Failed to evaluate prompt]";
    }
    n_past += n_prompt_tokens;

    // 3. Generation Loop
    std::vector<llama_token_data> candidates; // Reuse buffer for sampling
    candidates.reserve(llama_n_vocab(llama_get_model(chat_ctx_)));

    while (n_remain > 0) {
        // Sample the next token
        float *logits = llama_get_logits(chat_ctx_);
        int n_vocab = llama_n_vocab(llama_get_model(chat_ctx_));
        candidates.clear();
        for (llama_token token_id = 0; token_id < n_vocab; ++token_id) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

        // --- Sampling Strategy (Using simple Greedy for now) ---
        // TODO: Implement more sophisticated sampling (temp, top-p, top-k) for better quality
        const llama_token new_token_id = llama_sample_token_greedy(chat_ctx_, &candidates_p);

        // Check for EOS or context limit
        if (new_token_id == eos_token || n_past >= n_ctx) {
            break;
        }

        // Convert token to string piece and append
        // Using llama_token_to_piece is generally safer than deprecated llama_token_to_str
        response_text += llama_token_to_piece(chat_ctx_, new_token_id);

        // Evaluate the newly sampled token to update the context for the next iteration
        if (llama_eval(chat_ctx_, &new_token_id, 1, n_past, n_threads)) {
             std::cerr << "Error evaluating new token during generation." << std::endl;
             response_text += " [Error generating full response]";
             break; // Stop generation on error
        }
        n_past++;
        n_remain--;
    }

    return response_text;
}


} // namespace tldr
