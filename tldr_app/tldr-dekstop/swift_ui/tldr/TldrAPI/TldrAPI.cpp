//
//  TldrAPI.cpp
//  TldrAPI
//
//  Created by Manu Hegde on 4/23/25.
//

#include "TldrAPI.hpp"
#include <iostream>
#include <string>

// Include the actual C++ library header
#include "include/tldr_api.h"

// The actual implementation - no need for forwards or simplified stubs

extern "C" {

// Forward directly to the real C++ implementation
bool tldr_initializeSystem() {
    try {
        return tldr_cpp_api::initializeSystem();
    } catch (const std::exception& e) {
        std::cerr << "Exception in tldr_initializeSystem: " << e.what() << std::endl;
        return false;
    }
}

void tldr_cleanupSystem() {
    try {
        tldr_cpp_api::cleanupSystem();
    } catch (const std::exception& e) {
        std::cerr << "Exception in tldr_cleanupSystem: " << e.what() << std::endl;
    }
}

void tldr_addCorpus(const char* sourcePath) {
    if (sourcePath) {
        try {
            tldr_cpp_api::addCorpus(std::string(sourcePath));
        } catch (const std::exception& e) {
            std::cerr << "Exception in tldr_addCorpus: " << e.what() << std::endl;
        }
    }
}

void tldr_deleteCorpus(const char* corpusId) {
    if (corpusId) {
        try {
            tldr_cpp_api::deleteCorpus(std::string(corpusId));
        } catch (const std::exception& e) {
            std::cerr << "Exception in tldr_deleteCorpus: " << e.what() << std::endl;
        }
    }
}

void tldr_queryRag(const char* user_query, const char* embeddings_url, const char* chat_url) {
    try {
        tldr_cpp_api::queryRag(
            user_query ? std::string(user_query) : std::string(),
            embeddings_url ? std::string(embeddings_url) : std::string("http://localhost:8084/embeddings"),
            chat_url ? std::string(chat_url) : std::string("http://localhost:8088/v1/chat/completions")
        );
    } catch (const std::exception& e) {
        std::cerr << "Exception in tldr_queryRag: " << e.what() << std::endl;
    }
}

// Direct access to the test function
int tldr_api_trial_tldr() {
    std::cout << "tldr_api_trial_tldr called from Swift" << std::endl;
    try {
        bool result = tldr_cpp_api::initializeSystem();
        return result ? 42 : 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception in tldr_api_trial_tldr: " << e.what() << std::endl;
        return 0;
    }
}

} // extern "C"

int tldr_trial_main() {
    // Initialize system
    if (!tldr_initializeSystem()) {
        std::cerr << "Failed to initialize system" << std::endl;
        return 1;
    }

    std::cout << "System initialized successfully" << std::endl;

    // Cleanup system
    tldr_cleanupSystem();

    return 0;
}
