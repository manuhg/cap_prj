//
//  TldrAPI.cpp
//  TldrAPI
//
//  Created by Manu Hegde on 4/23/25.
//

#include "TldrAPI.hpp"
#include <iostream>
#include <string>

extern "C" {

// Implementation of function used by Swift - simplified version
int tldr_api_trial_tldr() {
    std::cout << "TldrAPI trial function called from Swift" << std::endl;
    return 42; // Just return a value to confirm it's working
}

// Simplified stubs for function calls from Swift to C++
bool tldr_initializeSystem() {
    std::cout << "tldr_initializeSystem called" << std::endl;
    return true;
}

void tldr_cleanupSystem() {
    std::cout << "tldr_cleanupSystem called" << std::endl;
}

void tldr_addCorpus(const char* sourcePath) {
    if (sourcePath) {
        std::cout << "tldr_addCorpus called with: " << sourcePath << std::endl;
    }
}

void tldr_deleteCorpus(const char* corpusId) {
    if (corpusId) {
        std::cout << "tldr_deleteCorpus called with: " << corpusId << std::endl;
    }
}

void tldr_queryRag(const char* user_query, const char* embeddings_url, const char* chat_url) {
    std::cout << "tldr_queryRag called with: " << std::endl;
    if (user_query) std::cout << "  query: " << user_query << std::endl;
    if (embeddings_url) std::cout << "  embeddings_url: " << embeddings_url << std::endl;
    if (chat_url) std::cout << "  chat_url: " << chat_url << std::endl;
}

} // extern "C"

// Implementation of TldrAPI static methods - simplified for testing
bool TldrAPI::initializeSystem() {
    std::cout << "TldrAPI::initializeSystem called" << std::endl;
    return true;
}

void TldrAPI::cleanupSystem() {
    std::cout << "TldrAPI::cleanupSystem called" << std::endl;
}

void TldrAPI::addCorpus(const std::string& sourcePath) {
    std::cout << "TldrAPI::addCorpus called with: " << sourcePath << std::endl;
}

void TldrAPI::deleteCorpus(const std::string& corpusId) {
    std::cout << "TldrAPI::deleteCorpus called with: " << corpusId << std::endl;
}

void TldrAPI::queryRag(const std::string& user_query, const std::string& embeddings_url, const std::string& chat_url) {
    std::cout << "TldrAPI::queryRag called with: " << std::endl;
    std::cout << "  query: " << user_query << std::endl;
    std::cout << "  embeddings_url: " << embeddings_url << std::endl;
    std::cout << "  chat_url: " << chat_url << std::endl;
}

int tldr_trial_main() {
    // Initialize system
    if (!TldrAPI::initializeSystem()) {
        std::cerr << "Failed to initialize system" << std::endl;
        return 1;
    }
    
    return 0;
}
