//
//  TldrAPI.cpp
//  TldrAPI
//
//  Created by Manu Hegde on 4/23/25.
//  Provides bridge implementation between C++ and Swift for TLDR functionality.
//

#include "TldrAPI.hpp" // Includes TldrAPI_C.h
#include "tldr_api.h"  // C++ API header

#include <string>
#include <iostream>
#include <cstring>

// Provide the C implementations that call the actual C++ library functions.
extern "C" {

// Test function that can be called from Swift
int tldr_api_trial_tldr(void) {
    std::cout << "[TldrAPI Wrapper] tldr_api_trial_tldr() called. Calling real library..." << std::endl;
    bool success = tldr_cpp_api::initializeSystem();
    if (!success) {
        std::cerr << "[TldrAPI Wrapper] Real library init failed in trial function!" << std::endl;
        return -1;
    }
    std::cout << "[TldrAPI Wrapper] Real library init succeeded in trial function." << std::endl;
    tldr_cpp_api::cleanupSystem();
    return 123;
}

// Initialize the TLDR system
bool tldr_initializeSystem(void) {
    std::cout << "[TldrAPI Wrapper] tldr_initializeSystem() called. Calling real library..." << std::endl;
    bool result = tldr_cpp_api::initializeSystem();
    std::cout << "[TldrAPI Wrapper] Real library initializeSystem() returned: " << result << std::endl;
    return result;
}

// Clean up the system
void tldr_cleanupSystem(void) {
    std::cout << "[TldrAPI Wrapper] tldr_cleanupSystem() called. Calling real library..." << std::endl;
    tldr_cpp_api::cleanupSystem();
    std::cout << "[TldrAPI Wrapper] Real library cleanupSystem() finished." << std::endl;
}

// Add a corpus from a PDF file or directory
void tldr_addCorpus(const char* sourcePath) {
    std::cout << "[TldrAPI Wrapper] tldr_addCorpus() called with path: " 
              << (sourcePath ? sourcePath : "NULL") << std::endl;
    if (sourcePath) {
        tldr_cpp_api::addCorpus(std::string(sourcePath));
    } else {
        std::cerr << "[TldrAPI Wrapper] Error: sourcePath is NULL." << std::endl;
    }
}

// Add a single PDF file to the corpus
void tldr_addFileToCorpus(const char* filePath) {
    std::cout << "[TldrAPI Wrapper] tldr_addFileToCorpus() called with path: " 
              << (filePath ? filePath : "NULL") << std::endl;
    if (filePath) {
        tldr_cpp_api::addFileToCorpus(std::string(filePath));
    } else {
        std::cerr << "[TldrAPI Wrapper] Error: filePath is NULL." << std::endl;
    }
}

// Delete a corpus by ID
void tldr_deleteCorpus(const char* corpusId) {
    std::cout << "[TldrAPI Wrapper] tldr_deleteCorpus() called with ID: " 
              << (corpusId ? corpusId : "NULL") << std::endl;
    if (corpusId) {
        tldr_cpp_api::deleteCorpus(std::string(corpusId));
    } else {
        std::cerr << "[TldrAPI Wrapper] Error: corpusId is NULL." << std::endl;
    }
}

// Query the RAG system
RagResult* tldr_queryRag(const char* user_query, const char* corpus_dir) {
    std::cout << "[TldrAPI Wrapper] tldr_queryRag() called with query: " 
              << (user_query ? user_query : "NULL") << std::endl;
    
    if (!user_query) {
        std::cerr << "[TldrAPI Wrapper] Error: user_query is NULL." << std::endl;
        return nullptr;
    }
    
    try {
        // Call the C++ API
        std::string corpusDir = corpus_dir ? std::string(corpus_dir) : "/Users/manu/proj_tldr/corpus/current/";
        auto result = tldr_cpp_api::queryRag(std::string(user_query), corpusDir);
        
        // Allocate memory for the result
        auto* c_result = new RagResult();
        
        // Copy the response string
        c_result->response = strdup(result.response.c_str());
        
        // Allocate memory for context chunks
        c_result->context_chunks_count = result.context_chunks.size();
        c_result->context_chunks = new ContextChunk[c_result->context_chunks_count];
        
        // Copy context chunks
        for (size_t i = 0; i < c_result->context_chunks_count; ++i) {
            const auto& chunk = result.context_chunks[i];
            c_result->context_chunks[i].text = strdup(std::get<0>(chunk).c_str());
            c_result->context_chunks[i].similarity = std::get<1>(chunk);
            c_result->context_chunks[i].hash = std::get<2>(chunk);
        }
        
        return c_result;
    } catch (const std::exception& e) {
        std::cerr << "[TldrAPI Wrapper] Exception in tldr_queryRag: " << e.what() << std::endl;
        return nullptr;
    }
}

// Free memory allocated by tldr_queryRag
void tldr_freeRagResult(RagResult* result) {
    if (!result) return;
    
    // Free response string
    if (result->response) {
        free((void*)result->response);
    }
    
    // Free context chunks
    if (result->context_chunks) {
        for (size_t i = 0; i < result->context_chunks_count; ++i) {
            if (result->context_chunks[i].text) {
                free((void*)result->context_chunks[i].text);
            }
        }
        delete[] result->context_chunks;
    }
    
    // Free the result structure itself
    delete result;
}

} // extern "C"
