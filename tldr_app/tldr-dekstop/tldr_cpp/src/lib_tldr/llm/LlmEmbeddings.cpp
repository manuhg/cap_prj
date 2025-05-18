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
#include <vector>
#include <vector>
#include <chrono>

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

    // run model
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

LlmEmbeddings::LlmEmbeddings(std::string model_path) {
    this->model_path = model_path;
    call_times_ms = std::vector<double>();
    batch_sizes = std::vector<size_t>();
    prompt_sizes = std::vector<size_t>();
}

bool LlmEmbeddings::initialize_model() {
    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();

    this->model = llama_model_load_from_file(model_path.c_str(), model_params);
    this->vocab = llama_model_get_vocab(model);

    std::cout<<"Embeddings initialized"<<std::endl;

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
    
    // Acquire a context from the pool
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
    
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    // tokenize the prompts and trim
    std::vector<std::vector<int32_t> > inputs;

    for (const auto &prompt: input_batch) {
        auto inp = common_tokenize(ctx, std::string(prompt), true, true);
        if (inp.size() > n_batch) {
            std::cerr <<
                    "number of tokens in input line (" << (long long int) inp.size() << ") exceeds batch size ("
                    << (long long int) n_batch << "), increase batch size and re-run: " << __func__ << std::endl;
            return std::vector<std::vector<float>>();
        }
        inputs.push_back(inp);
    }

    // check if the last token is SEP
    // it should be automatically added by the tokenizer when 'tokenizer.ggml.add_eos_token' is set to 'true'
    for (auto &inp: inputs) {
        if (inp.empty() || inp.back() != llama_vocab_sep(vocab)) {
            std::cerr << "warn: last token in the prompt is not SEP: " << __func__ << std::endl;
            std::cerr << "warn: 'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header: " << __func__
                    << std::endl;
        }
    }


    // initialize batch
    const int n_prompts = input_batch.size();
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // count number of embeddings
    int n_embd_count = 0;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
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
    float *out = emb + e * n_embd;
    batch_decode(ctx, batch, out, s, n_embd, params.embd_normalize);

    // clean up
    llama_batch_free(batch);

    auto call_end = std::chrono::high_resolution_clock::now();

    // convert to 2D vector
    std::vector<std::vector<float>> embeddings_vec;
    embeddings_vec.reserve(n_embd_count);  // Optional but more efficient

    for (size_t i = 0; i < n_embd_count; ++i) {
        embeddings_vec.emplace_back(
            embeddings.begin() + i * n_embd,
            embeddings.begin() + (i + 1) * n_embd
        );
    }

    double total_ms = std::chrono::duration<double, std::milli>(call_end - call_start).count();
    call_times_ms.push_back(total_ms);
    batch_sizes.push_back(input_batch.size());
    prompt_sizes.push_back(input_batch.empty()?0:input_batch[0].size());

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
