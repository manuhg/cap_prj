//
//  TldrAPI.cpp
//  TldrAPI
//
//  Created by Manu Hegde on 4/23/25.
//

#include "TldrAPI.hpp"
#include <iostream>
#include <string>

// Include the actual C++ library header with full path
#include "include/tldr_api.h"

// If the namespace is not being recognized, we can directly use the functions
// from the tldr library that's being linked with this code
namespace tldr_cpp_api {
    extern bool initializeSystem();
    extern void cleanupSystem();
    extern void addCorpus(const std::string& sourcePath);
    extern void deleteCorpus(const std::string& corpusId);
    extern void queryRag(const std::string& user_query, 
                         const std::string& embeddings_url,
                         const std::string& chat_url);
}

extern "C" {

// Implementation of function used by Swift - test function
int tldr_api_trial_tldr() {
    std::cout << "TldrAPI trial function called from Swift - using real C++ library" << std::endl;
    tldr_cpp_api::initializeSystem();
    return 42; // Just return a value to confirm it's working
}

// Forward calls to the actual C++ library
bool tldr_initializeSystem() {
    std::cout << "Forwarding to real initializeSystem implementation" << std::endl;
    return TldrAPI::wrapper_initializeSystem();
}

void tldr_cleanupSystem() {
    std::cout << "Forwarding to real cleanupSystem implementation" << std::endl;
    TldrAPI::cleanupSystem();
}

void tldr_addCorpus(const char* sourcePath) {
    if (sourcePath) {
        std::cout << "Forwarding addCorpus with path: " << sourcePath << std::endl;
        TldrAPI::addCorpus(std::string(sourcePath));
    }
}

void tldr_deleteCorpus(const char* corpusId) {
    if (corpusId) {
        std::cout << "Forwarding deleteCorpus with ID: " << corpusId << std::endl;
        TldrAPI::deleteCorpus(std::string(corpusId));
    }
}

void tldr_queryRag(const char* user_query, const char* embeddings_url, const char* chat_url) {
    std::cout << "Forwarding queryRag with query: " << (user_query ? user_query : "(null)") << std::endl;
    TldrAPI::queryRag(
        user_query ? std::string(user_query) : std::string(),
        embeddings_url ? std::string(embeddings_url) : std::string("http://localhost:8084/embeddings"),
        chat_url ? std::string(chat_url) : std::string("http://localhost:8088/v1/chat/completions")
    );
}

} // extern "C"

// These implementations are now just wrappers around the actual library functions
// which are defined in the tldr C++ library we're linking against
bool TldrAPI::wrapper_initializeSystem() {
    std::cout << "TldrAPI::initializeSystem called - Using real C++ implementation" << std::endl;
    try {
        // Call the real implementation from the tldr C++ library
        return tldr_cpp_api::initializeSystem();
    } catch (const std::exception& e) {
        std::cerr << "Error initializing system: " << e.what() << std::endl;
        return false;
    }
}

void TldrAPI::cleanupSystem() {
    std::cout << "TldrAPI::cleanupSystem called - Using real C++ implementation" << std::endl;
    try {
        // Call the real implementation from the tldr C++ library
        tldr_cpp_api::cleanupSystem();
    } catch (const std::exception& e) {
        std::cerr << "Error cleaning up system: " << e.what() << std::endl;
    }
}

void TldrAPI::addCorpus(const std::string& sourcePath) {
    std::cout << "TldrAPI::addCorpus called - Using real C++ implementation" << std::endl;
    std::cout << "Source path: " << sourcePath << std::endl;
    try {
        // Call the real implementation from the tldr C++ library
        tldr_cpp_api::addCorpus(sourcePath);
    } catch (const std::exception& e) {
        std::cerr << "Error adding corpus: " << e.what() << std::endl;
    }
}

void TldrAPI::deleteCorpus(const std::string& corpusId) {
    std::cout << "TldrAPI::deleteCorpus called - Using real C++ implementation" << std::endl;
    std::cout << "Corpus ID: " << corpusId << std::endl;
    try {
        // Call the real implementation from the tldr C++ library
        tldr_cpp_api::deleteCorpus(corpusId);
    } catch (const std::exception& e) {
        std::cerr << "Error deleting corpus: " << e.what() << std::endl;
    }
}

void TldrAPI::queryRag(const std::string& user_query, 
                       const std::string& embeddings_url,
                       const std::string& chat_url) {
    std::cout << "TldrAPI::queryRag called - Using real C++ implementation" << std::endl;
    std::cout << "Query: " << user_query << std::endl;
    std::cout << "Embeddings URL: " << embeddings_url << std::endl;
    std::cout << "Chat URL: " << chat_url << std::endl;
    try {
        // Call the real implementation from the tldr C++ library
        tldr_cpp_api::queryRag(user_query, embeddings_url, chat_url);
    } catch (const std::exception& e) {
        std::cerr << "Error querying RAG: " << e.what() << std::endl;
    }
}

int tldr_trial_main() {
    // Initialize system
    if (!TldrAPI::wrapper_initializeSystem()) {
        std::cerr << "Failed to initialize system" << std::endl;
        return 1;
    }
    
    return 0;
}
