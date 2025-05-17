#include "tldr_api.h"
#include "lib_tldr.h"
#include <iostream>

namespace tldr_cpp_api {

bool initializeSystem() {
    return ::initializeSystem();
}

void cleanupSystem() {
    ::cleanupSystem();
}

void addCorpus(const std::string& sourcePath) {
    ::addCorpus(sourcePath);
}

void deleteCorpus(const std::string& corpusId) {
    ::deleteCorpus(corpusId);
}

RagResult queryRag(const std::string& user_query, const std::string& corpus_dir) {
    // Call the global queryRag function
    auto result = ::queryRag(user_query, corpus_dir);
    
    // Convert to tldr_cpp_api::RagResult
    RagResult api_result;
    api_result.response = result.response;
    
    // Convert context chunks
    for (const auto& chunk : result.context_chunks) {
        api_result.context_chunks.emplace_back(
            chunk.text,
            chunk.similarity,
            chunk.hash
        );
    }
    
    return api_result;
}

} // namespace tldr_cpp_api

// C linkage function for Swift to call
extern "C" {
    int tldr_api_trial_tldr() {
        std::cout << "TldrAPI trial function called from Swift" << std::endl;
        tldr_cpp_api::initializeSystem();
        return 42;
    }
}