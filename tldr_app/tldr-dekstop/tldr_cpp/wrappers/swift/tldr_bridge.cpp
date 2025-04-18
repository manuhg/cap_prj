#include "tldr_bridge.h"
#include "../../src/tldr.h"
#include <string>
#include <cstring>

extern "C" {

bool tldr_initialize_system(void) {
    return tldr::initializeSystem();
}

void tldr_cleanup_system(void) {
    tldr::cleanupSystem();
}

void tldr_add_corpus(const char* source_path) {
    tldr::addCorpus(source_path);
}

void tldr_delete_corpus(const char* corpus_id) {
    tldr::deleteCorpus(corpus_id);
}

void tldr_query_rag(const char* user_query, 
                    const char* embeddings_url,
                    const char* chat_url) {
    tldr::queryRag(user_query, embeddings_url, chat_url);
}

char* tldr_translate_path(const char* path) {
    std::string result = tldr::translatePath(path);
    char* cstr = new char[result.length() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

} // extern "C" 