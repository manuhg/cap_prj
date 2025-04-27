#ifndef TLDR_CPP_TLDR_H
#define TLDR_CPP_TLDR_H

#include <string>
#include "include/tldr_api.h"

// Forward declaration of C++ functions
int tldr_trial_main();

// C-compatible function declarations for Swift interop
#ifdef __cplusplus
extern "C" {
#endif

// Test function that can be called from Swift
int tldr_api_trial_tldr();

// Returns 1 if initialization was successful, 0 otherwise
bool tldr_initializeSystem();

// Cleans up the system
void tldr_cleanupSystem();

// Adds a corpus from a PDF file path
void tldr_addCorpus(const char* sourcePath);

// Deletes a corpus by ID
void tldr_deleteCorpus(const char* corpusId);

// Queries the RAG system
void tldr_queryRag(const char* user_query, const char* embeddings_url = nullptr, const char* chat_url = nullptr);

#ifdef __cplusplus
}
#endif

#endif // TLDR_CPP_TLDR_H
