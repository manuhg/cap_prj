//
// Modified version of llama.cpp/examples/embedding.cpp/.h
//

#include "LlmEmbeddings.h"

#include "common.h"
#include "llama.h"

#include <ctime>
#include <algorithm>
#include <iostream>
#include <vector>
#include <vector>
#include <vector>

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
}

bool LlmEmbeddings::initialize_model() {
    common_init();
    this->params.model = this->model_path;
    params.embedding = true;
    // For non-causal models, batch size must be equal to ubatch size
    params.n_ubatch = params.n_batch;
    params.cpuparams.n_threads=4;

    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    common_init_result llama_init = common_init_from_params(params);

    this->model = llama_init.model.get();
    this->ctx = llama_init.context.get();

    if (model == NULL) {
        throw "unable to load embdedding model at " + this->model_path + "\n";
    }

    this->vocab = llama_model_get_vocab(model);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);


    if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        std::cerr << " computing embeddings in encoder-decoder models is not supported :" << __func__ << std::endl;
        throw "Computing embeddings in encoder-decoder models is not supported";
    }

    if (n_ctx > n_ctx_train) {
        std::cerr << "warning: model was trained on only " << n_ctx_train << " context tokens (" << n_ctx <<
                " specified): " << __func__ << std::endl;
    }
}

std::vector<std::vector<float>> LlmEmbeddings::llm_get_embeddings(std::vector<std::string> input_batch) {
    // max batch size
    const uint64_t n_batch = params.n_batch;

    if (params.n_batch >= params.n_ctx)
        GGML_ASSERT(params.n_batch >= params.n_ctx);

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

    // convert to 2D vector
    std::vector<std::vector<float>> embeddings_vec;
    embeddings_vec.reserve(n_embd_count);  // Optional but more efficient

    for (size_t i = 0; i < n_embd; ++i) {
        embeddings_vec.emplace_back(
            embeddings.begin() + i * n_embd,
            embeddings.begin() + (i + 1) * n_embd
        );
    }

    return embeddings_vec;
}

void LlmEmbeddings::embedding_cleanup() {
    if (ctx != nullptr) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model != nullptr) {
        llama_model_free(model);
        model = nullptr;
    }
}
