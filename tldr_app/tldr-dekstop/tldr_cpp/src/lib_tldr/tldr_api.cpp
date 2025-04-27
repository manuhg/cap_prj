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

void queryRag(const std::string& user_query, 
              const std::string& embeddings_url,
              const std::string& chat_url) {
    ::queryRag(user_query, embeddings_url, chat_url);
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