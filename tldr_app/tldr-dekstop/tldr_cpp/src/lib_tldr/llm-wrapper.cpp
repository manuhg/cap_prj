#include "llm-wrapper.h"
#include "llama.h" // Include the full definition needed by the cpp file

#include "common.h" // Include common.h for helpers like normalize

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm> // For std::copy
#include <memory> // For potential future use with strdup cleanup
#include <mutex> // For std::once_flag
#include <cstdlib> // For std::atexit

namespace tldr {

// --- Static Instance and Initialization ---

// Static instance, managed internally
static LlmManager g_llm_manager_instance;
static std::once_flag g_init_flag;

// Accessor implementation
LlmManager& get_llm_manager() {
    // Ensure initialization has happened (optional, depends on usage pattern)
    // std::call_once(g_init_flag, ...); // Could re-trigger init logic if needed
    return g_llm_manager_instance;
}

// Moved initialization function
void initialize_llm_manager_once(const std::string& chat_model_path, const std::string& embeddings_model_path) {
    std::call_once(g_init_flag, [&]() {
        // llama_backend_init(false); // Backend init should happen higher up, e.g., in lib_tldr main init
        // std::atexit(llama_backend_free); // Same for backend free

        // Create a default config
        LLMConfig default_config;
        // TODO: Potentially load config values from a file or settings
        default_config.n_gpu_layers = 100; // Or set based on detection/settings
        default_config.n_ctx = 2048;
        default_config.n_batch = 512;
        default_config.pooling_type = LLAMA_POOLING_TYPE_MEAN; // Or CLS depending on model

        std::cout << "Initializing chat model..." << std::endl;
        if (!g_llm_manager_instance.initialize_chat_model(chat_model_path, default_config)) {
            std::cerr << "Failed to initialize chat model." << std::endl;
            // Handle initialization failure
        }

        std::cout << "Initializing embeddings model..." << std::endl;
        if (!g_llm_manager_instance.initialize_embeddings_model(embeddings_model_path, default_config)) {
            std::cerr << "Failed to initialize embeddings model." << std::endl;
            // Handle initialization failure
        }
    });
}

// --- LlmManager Class Implementation ---

LlmManager::LlmManager() : chat_model_(nullptr), chat_ctx_(nullptr), embeddings_model_(nullptr), embeddings_ctx_(nullptr)
{
    llama_backend_init();
}

LlmManager::~LlmManager() {
    if (chat_ctx_) {
        llama_free(chat_ctx_);
        chat_ctx_ = nullptr;
    }
    if (chat_model_) {
        llama_model_free(chat_model_);
        chat_model_ = nullptr;
    }
    if (embeddings_ctx_) {
        llama_free(embeddings_ctx_);
        embeddings_ctx_ = nullptr;
    }
    if (embeddings_model_) {
        llama_model_free(embeddings_model_);
        embeddings_model_ = nullptr;
    }
    llama_backend_free();
}

bool LlmManager::initialize_chat_model(const std::string& model_path, const LLMConfig& config) {
    if (chat_model_) return true; // Already initialized

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = config.n_gpu_layers;
    chat_model_ = llama_model_load_from_file(model_path.c_str(), mparams); // Use non-deprecated
    if (!chat_model_) {
        fprintf(stderr, "Error: Failed to load chat model: %s\n", model_path.c_str());
        return false;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = config.n_ctx;
    cparams.n_batch = config.n_batch;
    cparams.logits_all = false; // Only need logits for the last token in generation
    cparams.embeddings = false; // Use plural 'embeddings'

    chat_ctx_ = llama_new_context_with_model(chat_model_, cparams); // Use non-deprecated
    if (!chat_ctx_) {
        llama_model_free(chat_model_);
        chat_model_ = nullptr;
        fprintf(stderr, "Error: Failed to create chat context.\n");
        return false;
    }
    std::cout << "Chat model initialized successfully." << std::endl;
    chat_config_ = config; // Store the config used
    return true; // Return true on success
}

bool LlmManager::initialize_embeddings_model(const std::string& model_path, const LLMConfig& config) {
    if (embeddings_model_) return true; // Already initialized

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = config.n_gpu_layers;
    embeddings_model_ = llama_model_load_from_file(model_path.c_str(), mparams); // Use non-deprecated
    if (!embeddings_model_) {
        fprintf(stderr, "Error: Failed to load embeddings model: %s\n", model_path.c_str());
        return false;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = config.n_ctx;
    cparams.n_batch = config.n_batch;
    cparams.logits_all = true; // Need logits for embedding extraction potentially
    cparams.embeddings = true; // Use plural 'embeddings'

    embeddings_ctx_ = llama_new_context_with_model(embeddings_model_, cparams); // Use non-deprecated
    if (!embeddings_ctx_) {
        llama_model_free(embeddings_model_);
        embeddings_model_ = nullptr;
        fprintf(stderr, "Error: Failed to create embeddings context.\n");
        return false;
    }
    std::cout << "Embeddings model initialized successfully." << std::endl;
    embeddings_config_ = config; // Store the config used
    return true; // Return true on success
}

std::vector<std::vector<float>> LlmManager::get_embeddings(const std::vector<std::string_view>& texts) {
    if (!embeddings_model_ || !embeddings_ctx_) {
        fprintf(stderr, "Embeddings model/context not initialized.\n");
        return {};
    }

    // Get model vocabulary
    const auto * vocab = llama_model_get_vocab(embeddings_model_);
    if (!vocab) {
        fprintf(stderr, "%s: Could not get vocabulary from embedding model\n", __func__);
        return {}; // Return empty vector on error
    }

    // Determine if BOS token should be added (based on vocab type - common heuristic)
    bool add_bos = llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM; // Add BOS for SentencePiece models

    // Context size
    const int n_ctx = llama_n_ctx(embeddings_ctx_);
    const int n_embd = llama_n_embd(embeddings_model_);
    const int n_batch_config = embeddings_config_.n_batch; // Use stored config batch size
    const auto pooling_type = embeddings_config_.pooling_type; // Use stored config pooling type

    std::vector<std::vector<float>> all_embeddings;
    all_embeddings.reserve(texts.size());

    std::vector<std::vector<llama_token>> tokenized_inputs;
    tokenized_inputs.reserve(texts.size());
    std::vector<int> n_tokens_for_input;
    n_tokens_for_input.reserve(texts.size());
    std::vector<llama_seq_id> seq_ids;
    seq_ids.reserve(texts.size());

    // Tokenize all inputs
    for (size_t i = 0; i < texts.size(); ++i) {
        // Pre-allocate a buffer for tokens. Size based on text length is a guess; might need adjustment.
        std::vector<llama_token> tokens(texts[i].length() + (add_bos ? 1 : 0));
        int n_tokens = llama_tokenize(
            vocab,
            texts[i].data(), texts[i].length(),
            tokens.data(), tokens.size(),
            add_bos,        // add_special (BOS)
            false         // parse_special
        );

        if (n_tokens < 0) {
            fprintf(stderr, "%s: llama_tokenize failed! needed %d, allocated %d\n", __func__, -n_tokens, (int)tokens.size());
            // Optionally, resize and retry if n_tokens < 0 indicates insufficient space
            continue;
        }
        tokens.resize(n_tokens); // Resize to actual number of tokens

        tokenized_inputs.push_back(std::move(tokens));
        n_tokens_for_input.push_back(n_tokens);
        seq_ids.push_back(i); // Assign unique sequence ID for each input
    }

    if (tokenized_inputs.empty()) {
        fprintf(stderr, "No valid inputs to process after tokenization.\n");
        return {};
    }

    // Initialize batch (allocate memory)
    const int max_sequences = std::min((int)tokenized_inputs.size(), n_batch_config); // Consider config batch size as max parallel sequences
    struct llama_batch batch = llama_batch_init(n_batch_config, 0, max_sequences); // n_tokens, embd (0=derive), n_seq_max

    int current_input_idx = 0;
    std::vector<int> batch_indices(tokenized_inputs.size()); // Map batch output back to original input index

    while (current_input_idx < (int)tokenized_inputs.size()) {
        batch.n_tokens = 0; // Reset token count for the new batch

        int batch_token_count = 0;
        int n_in_batch = 0;

        // Fill the batch
        int temp_input_idx = current_input_idx; // Use a temp index for this batch filling loop
        for (; temp_input_idx < (int)tokenized_inputs.size() && n_in_batch < max_sequences; ++temp_input_idx) {
            const auto& tokens = tokenized_inputs[temp_input_idx];
            if (batch.n_tokens + tokens.size() > n_batch_config) {
                // Batch is full in terms of total tokens
                break;
            }

            const llama_seq_id seq_id = seq_ids[temp_input_idx];

            // Add tokens to batch manually
            for (size_t k = 0; k < tokens.size(); ++k) {
                int pos = batch.n_tokens;
                batch.token[pos] = tokens[k];
                batch.pos[pos]   = batch_token_count + k; // Position within this specific sequence
                batch.seq_id[pos][0] = seq_id; // Assign to the first seq_id slot for this token
                batch.n_seq_id[pos] = 1;       // Indicate 1 seq_id is used for this token pos
                batch.logits[pos] = (k == tokens.size() - 1); // Set logits flag for the last token
                batch.n_tokens++;
            }

            batch_token_count += n_tokens_for_input[temp_input_idx]; // Track total tokens added across sequences
            batch_indices[n_in_batch] = temp_input_idx; // Store original index
            n_in_batch++;
        }
        current_input_idx = temp_input_idx; // Update the main index to where we left off

        if (batch.n_tokens == 0) {
            break;
        }

        // Decode the batch
        int decode_status = llama_decode(embeddings_ctx_, batch);
        if (decode_status != 0) {
            fprintf(stderr, "Warning: llama_decode failed with status %d\n", decode_status);
            // Potentially skip this batch or handle error
            // For now, continue processing other batches
            continue; 
        }

        // Extract embeddings for the sequences processed in this batch
        float * output = llama_get_embeddings(embeddings_ctx_);
        if (!output) {
             fprintf(stderr, "Error: llama_get_embeddings returned null after decode.\n");
             continue;
        }

        for (int i = 0; i < n_in_batch; ++i) {
            int original_idx = batch_indices[i];
            llama_seq_id seq_id = seq_ids[original_idx];
            int n_tokens = n_tokens_for_input[original_idx];

            // Find the start and end token positions for this sequence in the batch
            int token_start_pos = -1;
            int token_end_pos = -1;
            for(int k=0; k < batch.n_tokens; ++k) {
                bool found = false;
                for (int s_idx = 0; s_idx < batch.n_seq_id[k]; ++s_idx) {
                     if (batch.seq_id[k][s_idx] == seq_id) {
                         if (token_start_pos == -1) token_start_pos = k;
                         token_end_pos = k;
                         found = true;
                         break;
                     }
                }
            }

            if (token_start_pos == -1 || token_end_pos == -1) {
                fprintf(stderr, "Warning: Could not find tokens for seq_id %d in batch.\n", seq_id);
                continue;
            }
            // Ensure token_end_pos matches n_tokens expected (sanity check)
            // This might be off if batching logic isn't perfect
            // if (token_end_pos - token_start_pos + 1 != n_tokens) {
            //     fprintf(stderr, "Warning: Token count mismatch for seq_id %d. Expected %d, found %d.\n", seq_id, n_tokens, token_end_pos - token_start_pos + 1);
            // }


            std::vector<float> final_embedding(n_embd);
            bool success = false;

            // Calculate final embedding based on pooling strategy
            if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
                // Use embedding of the last token
                 if (batch.logits[token_end_pos]) { // Check if logits were requested (should be true)
                     std::copy(output + token_end_pos * n_embd, output + (token_end_pos + 1) * n_embd, final_embedding.begin());
                     success = true;
                 } else {
                    fprintf(stderr, "Warning: Logits not requested for last token of seq_id %d.\n", seq_id);
                 }
            } else if (pooling_type == LLAMA_POOLING_TYPE_MEAN) {
                // Calculate mean of embeddings for this sequence
                std::fill(final_embedding.begin(), final_embedding.end(), 0.0f);
                int count = 0;
                for (int k = token_start_pos; k <= token_end_pos; ++k) {
                    // Ensure this token actually belongs to the sequence (might not if batch packing complex)
                    bool belongs = false;
                    for(int s_idx = 0; s_idx < batch.n_seq_id[k]; ++s_idx) {
                         if (batch.seq_id[k][s_idx] == seq_id) {
                             belongs = true; break;
                         }
                    }
                    if (belongs) {
                        float * embd_k = output + k * n_embd;
                        for(int j=0; j<n_embd; ++j) {
                            final_embedding[j] += embd_k[j];
                        }
                        count++;
                    }
                }
                if (count > 0) {
                    for(int j=0; j<n_embd; ++j) {
                        final_embedding[j] /= count;
                    }
                    success = true;
                }
            } else if (pooling_type == LLAMA_POOLING_TYPE_CLS) {
                 // Get model vocabulary (needed for token_cls)
                 const auto * vocab = llama_model_get_vocab(embeddings_model_);
                 if (!vocab) {
                     fprintf(stderr, "Failed to get embeddings model vocabulary.\n");
                     return {}; // Return empty if vocab fails
                 }

                 // Get the CLS token ID if using CLS pooling
                 llama_token cls_token_id = -1; 
                 if (pooling_type == LLAMA_POOLING_TYPE_CLS) {
                     cls_token_id = llama_token_cls(vocab); // Use vocab here
                 }

                 // Find CLS token (assuming it's the first token if add_bos is true)
                 if (add_bos && tokenized_inputs[original_idx][0] == cls_token_id) {
                     std::copy(output + token_start_pos * n_embd, output + (token_start_pos + 1) * n_embd, final_embedding.begin());
                     success = true;
                 } else {
                     fprintf(stderr, "Warning: CLS pooling requested but CLS token not found or not first for seq_id %d.\n", seq_id);
                     // Fallback? Maybe mean? For now, fail.
                 }
            }
            // Add LLAMA_POOLING_TYPE_MAX if needed

            if (success) {
                all_embeddings.push_back(std::move(final_embedding));
            }
        } // end loop over sequences in batch
    } // end while loop over inputs

    llama_batch_free(batch); // This should handle freeing the seq_id arrays too

    return all_embeddings;
}

std::string LlmManager::get_chat_response(const std::string& context, const std::string& prompt) {
     if (!chat_model_ || !chat_ctx_) {
        fprintf(stderr, "Chat model/context not initialized.\n");
        return "Error: Chat model not ready.";
    }

    // Get model vocabulary
    const auto * vocab = llama_model_get_vocab(chat_model_);
    if (!vocab) {
        fprintf(stderr, "Failed to get chat model vocabulary.\n");
        return "Error: Vocabulary unavailable.";
    }

    const int n_ctx = llama_n_ctx(chat_ctx_);
    struct llama_batch batch = llama_batch_init(512, 0, 1); // Max tokens, embd (0=derive), n_seq_max=1

    // --- Tokenize the combined prompt ---
    std::string full_prompt = context.empty() ? prompt : context + "\n\n" + prompt;
    // Pre-allocate buffer. Max size is context size minus some safety margin.
    std::vector<llama_token> prompt_tokens(n_ctx);
    int n_prompt_tokens = llama_tokenize(
        vocab,
        full_prompt.c_str(), full_prompt.length(),
        prompt_tokens.data(), prompt_tokens.size(),
        true,         // add_special (BOS/EOS based on model)
        true          // parse_special (allow special tokens)
    );

    if (n_prompt_tokens < 0) {
        fprintf(stderr, "%s: Failed to tokenize prompt (needed %d, allocated %d)\n", __func__, -n_prompt_tokens, (int)prompt_tokens.size());
        // Handle error - potentially resize and retry, or return error
        llama_batch_free(batch);
        return "Error: Tokenization failed.";
    }
    prompt_tokens.resize(n_prompt_tokens); // Resize to actual number of tokens

    if ((int)prompt_tokens.size() > n_ctx - 4) { // Leave space for generation
        fprintf(stderr, "Prompt too long (%d tokens, max %d)\n", (int)prompt_tokens.size(), n_ctx - 4);
        llama_batch_free(batch);
        return "Error: Prompt too long.";
    }

    // --- Initial Prompt Decoding ---
    batch.n_tokens = 0;
    for (int i = 0; i < (int)prompt_tokens.size(); ++i) {
        batch.token[batch.n_tokens]  = prompt_tokens[i];
        batch.pos[batch.n_tokens]    = i;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.logits[batch.n_tokens] = (i == (int)prompt_tokens.size() - 1); // Only request logits for the last prompt token
        batch.n_tokens++;
    }

    if (llama_decode(chat_ctx_, batch) != 0) {
        fprintf(stderr, "Error: Initial llama_decode failed for prompt.\n");
        llama_batch_free(batch);
        return "Error: Prompt decoding failed.";
    }
    // --- End Initial Prompt Decoding ---

    // --- Initialize Sampler Chain ---
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * sampler_chain = llama_sampler_chain_init(sparams);
    // Add samplers based on chat_config_
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_k (chat_config_.top_k));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_p (chat_config_.top_p, 1)); // min_keep=1
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_temp  (chat_config_.temp));
    // TODO: Add repetition penalty sampler if needed
    // Add the final selection sampler (greedy)
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_greedy());
    // --- End Sampler Chain Initialization ---

    // Main generation loop
    std::string response_text = "";
    n_prompt_tokens = (int)prompt_tokens.size();
    llama_token new_token_id = 0;
    int n_decoded = 0;
    const int max_response_tokens = n_ctx - n_prompt_tokens - 4; // Safety margin

    while (n_decoded < max_response_tokens) {
        // --- Prepare batch for the *next* token ---
        batch.n_tokens = 0;
        batch.token[batch.n_tokens]  = new_token_id; // The token we just sampled
        batch.pos[batch.n_tokens]    = n_prompt_tokens + n_decoded -1; // Position of the new token
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.logits[batch.n_tokens] = true; // Request logits for the *next* prediction
        batch.n_tokens++;
        // --- End Batch Preparation ---

        // Decode the single token batch
        if (llama_decode(chat_ctx_, batch) != 0) {
            fprintf(stderr, "Error: llama_decode failed during generation.\n");
            break;
        }

        // --- Start Sampling (Using Sampler Chain) --- 
        // Get the logits for the last token added to the batch
        // The sampler chain will apply transformations (top-k, top-p, temp)
        // and the final sampler (greedy) will select the token.
        new_token_id = llama_sampler_sample(sampler_chain, chat_ctx_, batch.n_tokens - 1);
        llama_sampler_accept(sampler_chain, new_token_id); // Update sampler state
        // --- End Sampling ---

        // Check for End-of-Sequence (EOS) token
        if (new_token_id == llama_token_eos(vocab)) {
            break; // End of sequence
        }

        // Append token to response
        {
            char piece[32]; // Buffer for the token piece (adjust size if needed)
            int n_chars = llama_token_to_piece(vocab, new_token_id, piece, sizeof(piece), 0, false); // lstrip=0, special=false
            if (n_chars < 0) {
                fprintf(stderr, "Error: llama_token_to_piece failed for token %d\n", new_token_id);
                // Handle error appropriately, maybe break or return
            } else {
                response_text.append(piece, n_chars);
            }
        }
        n_decoded++;

        // Update the prompt tokens for the next iteration - REMOVED, rely on KV cache
        // prompt_tokens.push_back(new_token_id);
        // prompt_tokens.erase(prompt_tokens.begin());
    }

    llama_sampler_free(sampler_chain); // Free the sampler chain
    llama_batch_free(batch);
    return response_text;
}

} // namespace tldr
