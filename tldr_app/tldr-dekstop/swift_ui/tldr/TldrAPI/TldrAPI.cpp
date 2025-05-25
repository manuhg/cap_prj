//
//  TldrAPI.cpp
//  TldrAPI
//
//  Created by Manu Hegde on 4/23/25.
//  Provides bridge implementation between C++ and Swift for TLDR functionality.
//

#include "TldrAPI.hpp"
#include "tldr_api.h"

// Core Foundation includes for macOS/iOS bundle access
#include <CoreFoundation/CoreFoundation.h>
#include <CoreServices/CoreServices.h> // For UInt8 type
#include <limits.h> // For PATH_MAX

#include <string>
#include <iostream>
#include <cstring>
#include <stdexcept> // For std::runtime_error

// Helper function to get the path to a resource in the app bundle
static std::string getResourcePath(const std::string& filename) {
    CFBundleRef mainBundle = CFBundleGetMainBundle();
    if (!mainBundle) {
        throw std::runtime_error("Could not get main bundle");
    }

    CFStringRef filenameStr = CFStringCreateWithCString(kCFAllocatorDefault, 
                                                      filename.c_str(), 
                                                      kCFStringEncodingUTF8);
    if (!filenameStr) {
        throw std::runtime_error("Could not create CFString from filename");
    }

    CFURLRef resourceURL = CFBundleCopyResourceURL(mainBundle, filenameStr, NULL, NULL);
    CFRelease(filenameStr);
    
    if (!resourceURL) {
        throw std::runtime_error("Could not find resource: " + filename);
    }

    char path[PATH_MAX];
    if (!CFURLGetFileSystemRepresentation(resourceURL, true, (UInt8*)path, PATH_MAX)) {
        CFRelease(resourceURL);
        throw std::runtime_error("Could not get filesystem path for resource");
    }

    std::string result(path);
    CFRelease(resourceURL);
    return result;
}

// Provide the C implementations that call the actual C++ library functions.
extern "C" {

// Initialize the TLDR system with model paths
bool tldr_initializeSystem(const char* chatModel, const char* embeddingsModel) {
    try {
        // Get paths to model files in the bundle
        std::string chatModelPath = getResourcePath(chatModel);
        std::string embeddingsModelPath = getResourcePath(embeddingsModel);
        std::cout << "Model paths obtained are as follows: \nchatModelPath:" << chatModelPath << "\nembeddingsModelPath:" << embeddingsModelPath << std::endl;

        // Initialize the LLM manager with the paths
        tldr_cpp_api::initializeSystem(chatModelPath.c_str(), embeddingsModelPath.c_str());
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing LLM models: " << e.what() << std::endl;
        return false;
    }
}

// Clean up the system
void tldr_cleanupSystem(void) {
    tldr_cpp_api::cleanupSystem();
}

// Add a corpus from a PDF file or directory
void tldr_addCorpus(const char* sourcePath) {
    tldr_cpp_api::addCorpus(sourcePath);
}

// Delete a corpus by ID
void tldr_deleteCorpus(const char* corpusId) {
    tldr_cpp_api::deleteCorpus(corpusId);
}

// Query the RAG system
RagResultC* tldr_queryRag(const char* user_query, const char* corpus_dir) {
    auto cpp_result = tldr_cpp_api::queryRag(user_query, corpus_dir);

    RagResultC* c_result = new RagResultC;
    c_result->response = strdup(cpp_result.response.c_str());
    c_result->referenced_document_count = cpp_result.referenced_document_count;
    c_result->context_chunks_count = cpp_result.context_chunks.size();
    c_result->context_chunks = new CtxChunkMetaC[c_result->context_chunks_count];

    for (size_t i = 0; i < c_result->context_chunks_count; ++i) {
        const auto& chunk = cpp_result.context_chunks[i];
        c_result->context_chunks[i].text = strdup(chunk.text.c_str());
        c_result->context_chunks[i].file_path = strdup(chunk.file_path.c_str());
        c_result->context_chunks[i].file_name = strdup(chunk.file_name.c_str());
        c_result->context_chunks[i].title = strdup(chunk.title.c_str());
        c_result->context_chunks[i].author = strdup(chunk.author.c_str());
        c_result->context_chunks[i].page_count = chunk.page_count;
        c_result->context_chunks[i].page_number = chunk.page_number;
        c_result->context_chunks[i].similarity = chunk.similarity;
        c_result->context_chunks[i].hash = chunk.hash;
    }
    return c_result;
}

void tldr_freeRagResult(RagResultC* result) {
    if (!result) return;
    free(result->response);
    for (size_t i = 0; i < result->context_chunks_count; ++i) {
        free(result->context_chunks[i].text);
        free(result->context_chunks[i].file_path);
        free(result->context_chunks[i].file_name);
        free(result->context_chunks[i].title);
        free(result->context_chunks[i].author);
    }
    delete[] result->context_chunks;
    delete result;
}

void tldr_freeString(char* str) {
    free(str);
}

} // extern "C"
