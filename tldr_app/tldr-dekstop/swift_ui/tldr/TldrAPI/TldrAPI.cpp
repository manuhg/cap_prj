//
//  TldrAPI.cpp (Simplified Stub)
//  TldrAPI
//
//  Created by Manu Hegde on 4/23/25.
//  Provides a stub implementation for Swift UI development.
//

#include "TldrAPI.hpp" // Includes TldrAPI_C.h indirectly

// Include the header from the actual tldr_cpp library
// Note: The path to this header must be added to Xcode's Header Search Paths.
#include "tldr_api.h"

#include <string>
#include <iostream>

// Provide the C implementations that call the actual C++ library functions.
extern "C" {

// Test function that can be called from Swift
int tldr_api_trial_tldr(void) {
    // Assuming the real library might not have this exact test function,
    // we can return a specific value or call another simple function.
    std::cout << "[TldrAPI Wrapper] tldr_api_trial_tldr() called. Calling real library..." << std::endl;
    // Replace with an actual call if one exists, or keep simple.
    bool success = tldr_cpp_api::initializeSystem(); // Example call to check linking (use correct namespace)
    if (!success) {
        std::cerr << "[TldrAPI Wrapper] Real library init failed in trial function!" << std::endl;
        return -1; // Indicate failure
    }
    std::cout << "[TldrAPI Wrapper] Real library init succeeded in trial function." << std::endl;
    tldr_cpp_api::cleanupSystem(); // Clean up after test init (use correct namespace)
    return 123; // Return a distinct value for testing
}

// Returns 1 if initialization was successful, 0 otherwise
bool tldr_initializeSystem(void) {
    std::cout << "[TldrAPI Wrapper] tldr_initializeSystem() called. Calling real library..." << std::endl;
    bool result = tldr_cpp_api::initializeSystem(); // Use correct namespace
    std::cout << "[TldrAPI Wrapper] Real library initializeSystem() returned: " << result << std::endl;
    return result;
}

// Cleans up the system
void tldr_cleanupSystem(void) {
    std::cout << "[TldrAPI Wrapper] tldr_cleanupSystem() called. Calling real library..." << std::endl;
    tldr_cpp_api::cleanupSystem(); // Use correct namespace
    std::cout << "[TldrAPI Wrapper] Real library cleanupSystem() finished." << std::endl;
}

// Adds a corpus from a PDF file path
void tldr_addCorpus(const char* sourcePath) {
    std::cout << "[TldrAPI Wrapper] tldr_addCorpus() called with path: " << (sourcePath ? sourcePath : "NULL") << ". Calling real library..." << std::endl;
    if (sourcePath) {
        tldr_cpp_api::addCorpus(std::string(sourcePath)); // Use correct namespace
        std::cout << "[TldrAPI Wrapper] Real library addCorpus() finished." << std::endl;
    } else {
        std::cerr << "[TldrAPI Wrapper] Error: sourcePath is NULL." << std::endl;
    }
}

// Deletes a corpus by ID
void tldr_deleteCorpus(const char* corpusId) {
     std::cout << "[TldrAPI Wrapper] tldr_deleteCorpus() called with ID: " << (corpusId ? corpusId : "NULL") << ". Calling real library..." << std::endl;
    if (corpusId) {
        tldr_cpp_api::deleteCorpus(std::string(corpusId)); // Use correct namespace
         std::cout << "[TldrAPI Wrapper] Real library deleteCorpus() finished." << std::endl;
    } else {
         std::cerr << "[TldrAPI Wrapper] Error: corpusId is NULL." << std::endl;
    }
}

// Queries the RAG system
void tldr_queryRag(const char* user_query, const char* embeddings_url, const char* chat_url) {
    std::cout << "[TldrAPI Wrapper] tldr_queryRag() called with query: " << (user_query ? user_query : "NULL") << ". Calling real library..." << std::endl;
    if (user_query && embeddings_url && chat_url) {
        tldr_cpp_api::queryRag(std::string(user_query), std::string(embeddings_url), std::string(chat_url)); // Use correct namespace
         std::cout << "[TldrAPI Wrapper] Real library queryRag() finished." << std::endl;
    } else {
        std::cerr << "[TldrAPI Wrapper] Error: One or more arguments to queryRag are NULL." << std::endl;
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
