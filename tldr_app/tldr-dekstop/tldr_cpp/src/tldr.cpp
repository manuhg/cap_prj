#include "tldr.h"
#include "lib_tldr/lib_tldr.h"

namespace tldr {

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