//
// Modified version of llama.cpp/examples/simple.cpp
//

#ifndef LLM_CHAT_H
#define LLM_CHAT_H

#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct llm_result {
    bool error;
    std::string error_message;
    std::string result;
};

class llm_chat {
    llm_result chat_with_llm(std::string prompt, std::string model_path,int ngl = 99,int n_predict = 32);
};

#endif //LLM_CHAT_H
