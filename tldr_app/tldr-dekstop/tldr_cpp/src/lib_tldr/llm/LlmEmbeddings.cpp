//
// Modified version of llama.cpp/examples/embedding.cpp/.h
//

#include "LlmEmbeddings.h"

#include "common.h"
#include "llama.h"
#include "arg.h"
#include "../constants.h"

#include <ctime>
#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


static void batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, {seq_id}, true);
    }
}

static void batch_decode(llama_context *ctx, llama_batch &batch, float *output, int n_seq, int n_embd, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const struct llama_model *model = llama_get_model(ctx);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
        // encoder-only model
        if (llama_encode(ctx, batch) < 0) {
            std::cerr << "failed to encode :" << __func__ << std::endl;
        }
    } else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        // decoder-only model
        if (llama_decode(ctx, batch) < 0) {
            std::cerr << "failed to decode :" << __func__ << std::endl;
        }
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float *embd = nullptr;
        int embd_pos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // try to get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            GGML_ASSERT(embd != NULL && "failed to get token embeddings");
        } else {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
            GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
        }

        float *out = output + embd_pos * n_embd;
        common_embd_normalize(embd, out, n_embd, embd_norm);
    }
}

LlmEmbeddings::LlmEmbeddings() {
    call_times_ms = std::vector<double>();
    batch_sizes = std::vector<size_t>();
    prompt_sizes = std::vector<size_t>();
}

bool LlmEmbeddings::initialize_model(const std::string& model_path) {
    this->model_path = model_path; 
    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();

    this->model = llama_model_load_from_file(model_path.c_str(), model_params);
    this->vocab = llama_model_get_vocab(model);

    // Set number of OpenMP threads
    int max_threads = omp_get_max_threads();
    std::cout << "Embeddings initialized with " << max_threads << " OpenMP threads available" << std::endl;

    // Create context parameters for embeddings
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ubatch = 2048;
    ctx_params.embeddings = true;
    
    // Create context pool with sizes defined in constants.h
    context_pool = std::make_unique<tldr::LlmContextPool>(model, EMBEDDING_MIN_CONTEXTS, EMBEDDING_MAX_CONTEXTS, ctx_params);
    
    return true;
}

std::vector<std::vector<float>> LlmEmbeddings::llm_get_embeddings(std::vector<std::string_view> input_batch) {
    // std::cout<<"Embeddings input batch size:"<<input_batch.size()<<"x"<<input_batch[0].size() <<std::endl;
    // max batch size
    const uint64_t n_batch = params.n_batch;
    auto call_start = std::chrono::high_resolution_clock::now();
    
    // We'll determine if we need multiple contexts based on input size
    // For small batches, a single context is sufficient
    // For large batches, we'll use multiple contexts to parallelize further
    const bool use_multiple_contexts = input_batch.size() > EMBEDDING_MIN_CONTEXTS * 2;
    
    // Default to single context first
    auto ctx_handle = context_pool->acquire_context();
    if (!ctx_handle) {
        std::cerr << "Failed to acquire context from pool" << std::endl;
        return std::vector<std::vector<float>>();
    }
    
    llama_context *ctx = ctx_handle->get();
    if (ctx == NULL) {
        std::cerr << "Acquired null context from pool" << std::endl;
        return std::vector<std::vector<float>>();
    }

    const struct llama_model *model = llama_get_model(ctx);
    enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    std::vector<std::vector<int32_t>> inputs;
    inputs.resize(input_batch.size());

    // Tokenize the inputs in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int) input_batch.size(); i++) {
        std::string inp = input_batch[i].data();
        
        // First get token count (returns negative count when only measuring)
        const int n_tokens = -llama_tokenize(vocab, inp.c_str(), inp.size(), NULL, 0, true, true);
        
        if (n_tokens <= 0) {
            continue; // Skip this input
        }
        
        // Allocate space and get the actual tokens
        inputs[i].resize(n_tokens);
        if (llama_tokenize(vocab, inp.c_str(), inp.size(), inputs[i].data(), inputs[i].size(), true, true) < 0) {
            inputs[i].clear(); // Mark as failed
        }
    }
    
    // Check if any tokenization failed
    bool tokenization_failed = false;
    for (const auto& tokens : inputs) {
        if (tokens.empty()) {
            tokenization_failed = true;
            break;
        }
    }
    
    if (tokenization_failed) {
        return std::vector<std::vector<float>>();
    }

    // check if the last token is SEP in parallel
    // it should be automatically added by the tokenizer when 'tokenizer.ggml.add_eos_token' is set to 'true'
    bool missing_sep = false;
    #pragma omp parallel for reduction(||:missing_sep)
    for (size_t i = 0; i < inputs.size(); i++) {
        auto &inp = inputs[i];
        missing_sep = (inp.empty() || inp.back() != llama_vocab_sep(vocab));
    }
    
    if (missing_sep) {
        std::cerr << "warn: last token in at least one prompt is not SEP: " << __func__ << std::endl;
        std::cerr << "warn: 'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header: " << __func__ << std::endl;
    }


    // initialize batch
    const int n_prompts = input_batch.size();
    
    // count number of embeddings in parallel
    int n_embd_count = 0;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        #pragma omp parallel for reduction(+:n_embd_count)
        for (int k = 0; k < n_prompts; k++) {
            n_embd_count += inputs[k].size();
        }
    } else {
        n_embd_count = n_prompts;
    }

    // allocate output
    const int n_embd = llama_model_n_embd(model);
    std::vector<float> embeddings(n_embd_count * n_embd, 0);
    float *emb = embeddings.data();
    
    if (use_multiple_contexts && n_prompts > 1) {
        // Multi-context approach for larger batches
        // We'll split the work across multiple contexts for parallel processing
        
        // Determine how many contexts to use
        const int max_contexts = std::min(EMBEDDING_MAX_CONTEXTS, (int)input_batch.size() / 2);
        const int contexts_to_use = std::min(max_contexts, omp_get_max_threads());
        
        // Only proceed with multi-context if we can get at least 2 contexts
        if (contexts_to_use >= 2) {
            std::vector<std::shared_ptr<tldr::ContextHandle>> context_handles;
            std::vector<llama_context*> contexts;
            
            // Store the first context we already acquired
            context_handles.push_back(std::move(ctx_handle));
            contexts.push_back(ctx);
            
            // Acquire additional contexts
            for (int c = 1; c < contexts_to_use; c++) {
                auto additional_handle = context_pool->acquire_context();
                if (!additional_handle || !additional_handle->get()) {
                    // Failed to get enough contexts, we'll use what we have
                    break;
                }
                context_handles.push_back(std::move(additional_handle));
                contexts.push_back(context_handles.back()->get());
            }
            
            // Divide work among contexts
            const int actual_contexts = contexts.size();
            const int prompts_per_context = (n_prompts + actual_contexts - 1) / actual_contexts;
            
            // Track embedding positions
            std::vector<int> embedding_offsets(actual_contexts + 1, 0);
            
            #pragma omp parallel for num_threads(actual_contexts)
            for (int c = 0; c < actual_contexts; c++) {
                // Calculate range for this context
                const int start_prompt = c * prompts_per_context;
                const int end_prompt = std::min(start_prompt + prompts_per_context, n_prompts);
                
                if (start_prompt >= end_prompt) continue;
                
                // Initialize batch for this context
                struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
                
                // Calculate embedding offset
                int local_embd_count = 0;
                if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
                    for (int k = start_prompt; k < end_prompt; k++) {
                        local_embd_count += inputs[k].size();
                    }
                } else {
                    local_embd_count = end_prompt - start_prompt;
                }
                
                #pragma omp critical
                {
                    embedding_offsets[c+1] = embedding_offsets[c] + local_embd_count;
                }
                
                // Process batches with this context
                int e = 0; // local embeddings count
                int s = 0; // local sequence count
                
                for (int k = start_prompt; k < end_prompt; k++) {
                    auto &inp = inputs[k];
                    const uint64_t n_toks = inp.size();
                    
                    // Encode if at capacity
                    if (batch.n_tokens + n_toks > n_batch) {
                        float *out = emb + (embedding_offsets[c] + e) * n_embd;
                        batch_decode(contexts[c], batch, out, s, n_embd, params.embd_normalize);
                        e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
                        s = 0;
                        common_batch_clear(batch);
                    }
                    
                    // Add to batch
                    batch_add_seq(batch, inp, s);
                    s += 1;
                }
                
                // Process final batch for this context
                if (batch.n_tokens > 0) {
                    float *out = emb + (embedding_offsets[c] + e) * n_embd;
                    batch_decode(contexts[c], batch, out, s, n_embd, params.embd_normalize);
                }
                
                // Clean up batch
                llama_batch_free(batch);
            }
            
            // All contexts have been used and can be returned to the pool automatically
            // via RAII when context_handles goes out of scope
            
            // Skip the single-context path since we've processed everything
            goto skip_single_context;
        }
    }
    
    // Single context path (fallback or when multiple contexts aren't needed)
    {
        struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
        
        // break into batches
        int e = 0; // number of embeddings already stored
        int s = 0; // number of prompts in current batch
        for (int k = 0; k < n_prompts; k++) {
            // clamp to n_batch tokens
            auto &inp = inputs[k];
            const uint64_t n_toks = inp.size();
            
            // encode if at capacity
            if (batch.n_tokens + n_toks > n_batch) {
                float *out = emb + e * n_embd;
                batch_decode(ctx, batch, out, s, n_embd, params.embd_normalize);
                e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
                s = 0;
                common_batch_clear(batch);
            }
            
            // add to batch
            batch_add_seq(batch, inp, s);
            s += 1;
        }
        
        // final batch
        if (batch.n_tokens > 0) {
            float *out = emb + e * n_embd;
            batch_decode(ctx, batch, out, s, n_embd, params.embd_normalize);
        }
        
        // clean up
        llama_batch_free(batch);
    }
    
    skip_single_context:
    // No additional cleanup needed here - batches are freed in their respective code paths

    auto call_end = std::chrono::high_resolution_clock::now();

    // convert to 2D vector in parallel
    std::vector<std::vector<float>> embeddings_vec;
    embeddings_vec.resize(n_embd_count);  // Pre-allocate to avoid race conditions

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_embd_count; ++i) {
        embeddings_vec[i].assign(
            embeddings.begin() + i * n_embd,
            embeddings.begin() + (i + 1) * n_embd
        );
    }

    double total_ms = std::chrono::duration<double, std::milli>(call_end - call_start).count();
    
    // Log performance information
    #pragma omp critical
    {
        call_times_ms.push_back(total_ms);
        batch_sizes.push_back(input_batch.size());
        prompt_sizes.push_back(input_batch.empty()?0:input_batch[0].size());
        
        // Optional: Print thread info for debugging
        // std::cout << "Processed batch of " << input_batch.size() << " items using " << omp_get_num_threads() << " threads in " << total_ms << "ms" << std::endl;
    }

    return embeddings_vec;
}

void LlmEmbeddings::embedding_cleanup() {
    // Clean up the context pool first
    if (context_pool) {
        context_pool->clear();
        context_pool.reset();
    }
    
    // Then free the model
    if (model != nullptr) {
        llama_model_free(model);
        model = nullptr;
    }
    if (!call_times_ms.empty()) {
        double total_sum = 0;
        for (double v : call_times_ms) total_sum += v;
        // helper lambda for median
        auto median=[](std::vector<double> v){ if(v.empty()) return 0.0; std::sort(v.begin(),v.end()); size_t mid=v.size()/2; return v.size()%2? v[mid]: (v[mid-1]+v[mid])/2.0;};
        auto median_size=[&](std::vector<size_t> v){ if(v.empty()) return 0.0; std::sort(v.begin(),v.end()); size_t mid=v.size()/2; return v.size()%2? (double)v[mid]: ((double)v[mid-1]+v[mid])/2.0;};
        double total_med=median(call_times_ms);
        double batch_med=median_size(batch_sizes);
        double prompt_med=median_size(prompt_sizes);
        std::cout << "Embedding stats across " << call_times_ms.size() << " calls: total time "
                  << total_sum/1000.0 << " s" << std::endl;
        std::cout << "Median call time "<< total_med/1000.0 << " s, median batch "<< batch_med
                  << ", median prompt size "<< prompt_med << std::endl;
    }
}
