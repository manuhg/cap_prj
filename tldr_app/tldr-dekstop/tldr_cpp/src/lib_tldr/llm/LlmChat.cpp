//
// Modified version of llama.cpp/examples/simple.cpp/.h
//

#include "LlmChat.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include "../constants.h"

LlmChat::LlmChat(std::string model_path) {
    this->model_path = model_path;
    call_times_ms = std::vector<double>();
    prompt_sizes = std::vector<size_t>();
}

void LlmChat::llm_chat_cleanup() {
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
        double total_sum=0; for(double v:call_times_ms) total_sum+=v;
        auto median=[&](std::vector<double> v){ std::sort(v.begin(),v.end()); size_t m=v.size()/2; return v.size()%2? v[m]:(v[m-1]+v[m])/2.0;};
        auto median_size=[&](std::vector<size_t> v){ std::sort(v.begin(),v.end()); size_t m=v.size()/2; return v.size()%2? (double)v[m]:((double)v[m-1]+v[m])/2.0;};
        double med=median(call_times_ms);
        double prompt_med=median_size(prompt_sizes);
        std::cout<<"Chat stats across "<<call_times_ms.size()<<" calls: total "<<total_sum/1000.0<<" s, median "<<med/1000.0<<" s, median prompt size "<<prompt_med<<std::endl;
    }
}

bool LlmChat::initialize_model() {
    try {
        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();

        this->model = llama_model_load_from_file(model_path.c_str(), model_params);
        this->vocab = llama_model_get_vocab(model);
        
        // Create context pool with sizes defined in constants.h
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048; // Default context size
        ctx_params.n_batch = 512; // Default batch size
        
        context_pool = std::make_unique<tldr::LlmContextPool>(model, CHAT_MIN_CONTEXTS, CHAT_MAX_CONTEXTS, ctx_params);
        
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Exception initializing chat model: " << e.what() << std::endl;
        return false;
    }
}

llm_result LlmChat::chat_with_llm(std::string prompt) {
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return {true, "unable to load model\n"};
    }

    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true,
                       true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return {true, "failed to tokenize the prompt\n"};
    }

    // Acquire a context from the pool
    auto ctx_handle = context_pool->acquire_context();
    if (!ctx_handle) {
        fprintf(stderr, "%s: error: failed to acquire context from pool\n", __func__);
        return {true, "failed to acquire context from pool\n"};
    }
    
    llama_context *ctx = ctx_handle->get();
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: acquired null context from pool\n", __func__);
        return {true, "acquired null context from pool\n"};
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler *smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    for (auto id: prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return {true, "failed to convert token to piece\n"};
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;
    std::string output = "";

    auto call_start = std::chrono::high_resolution_clock::now();

    // Get the context size from the context parameters
    const int ctx_size = llama_n_ctx(ctx);
    for (int n_pos = 0; n_pos + batch.n_tokens < ctx_size;) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return {true, "failed to eval\n"};
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return {true, "failed to convert token to piece\n"};;
            }
            std::string s(buf, n);
            output += s;
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f,
            n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    // Free the sampler but don't free the context - it will be returned to the pool
    llama_sampler_free(smpl);
    // Context is automatically returned to the pool when ctx_handle goes out of scope

    auto call_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(call_end - call_start).count();
    call_times_ms.push_back(total_ms);
    prompt_sizes.push_back(prompt.size());

    return {false, "", output};
}
