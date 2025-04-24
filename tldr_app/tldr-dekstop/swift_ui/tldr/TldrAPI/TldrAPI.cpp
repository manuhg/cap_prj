//
//  TldrAPI.cpp
//  TldrAPI
//
//  Created by Manu Hegde on 4/23/25.
//

#include <iostream>
#include "TldrAPI.hpp"
#include "TldrAPIPriv.hpp"

extern "C" {

bool tldr_initializeSystem() {
    return TldrAPI::initializeSystem();
}

void tldr_cleanupSystem() {
    TldrAPI::cleanupSystem();
}

void tldr_addCorpus(const char* sourcePath) {
    if (sourcePath) {
        TldrAPI::addCorpus(std::string(sourcePath));
    }
}

void tldr_deleteCorpus(const char* corpusId) {
    if (corpusId) {
        TldrAPI::deleteCorpus(std::string(corpusId));
    }
}

void tldr_queryRag(const char* user_query, const char* embeddings_url, const char* chat_url) {
    TldrAPI::queryRag(
        user_query ? std::string(user_query) : std::string(),
        embeddings_url ? std::string(embeddings_url) : std::string(),
        chat_url ? std::string(chat_url) : std::string()
    );
}


} extern "C"


int main() {
    // Initialize system
    if (!TldrAPI::initializeSystem()) {
        std::cerr << "Failed to initialize system" << std::endl;
        return 1;
    }

    std::string testFile = "~/proj_tldr/corpus/current/0.System Design Interview An Insider’s Guide by Alex Xu.pdf";

    TldrAPI::addCorpus(testFile);
    TldrAPI::queryRag("What does the book say about hotspot problem?");

    // Cleanup system
    TldrAPI::cleanupSystem();

    return 0;
}
