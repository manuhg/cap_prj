#include "tldr_api.h"
#include "lib_tldr.h"

namespace TldrAPI {

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

} // namespace tldr 