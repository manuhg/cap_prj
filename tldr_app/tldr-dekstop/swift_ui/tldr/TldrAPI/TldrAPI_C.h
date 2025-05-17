#ifndef TLDRAPI_C_H
#define TLDRAPI_C_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of RagResult structure
typedef struct RagResult RagResult;

// Structure to hold a context chunk
typedef struct {
    const char* text;
    float similarity;
    uint64_t hash;
} ContextChunk;

// Structure to hold RAG query results
struct RagResult {
    char* response;
    ContextChunk* context_chunks;
    size_t context_chunks_count;
};

// Test function that can be called from Swift
int tldr_api_trial_tldr(void);

// Initialize the TLDR system
bool tldr_initializeSystem(void);

// Clean up the system
void tldr_cleanupSystem(void);

// Add a corpus from a PDF file or directory
void tldr_addCorpus(const char* sourcePath);

// Add a single PDF file to the corpus
void tldr_addFileToCorpus(const char* filePath);

// Delete a corpus by ID
void tldr_deleteCorpus(const char* corpusId);

// Query the RAG system
RagResult* tldr_queryRag(const char* user_query, const char* corpus_dir);

// Free memory allocated by tldr_queryRag
void tldr_freeRagResult(RagResult* result);

#ifdef __cplusplus
}
#endif

#endif /* TLDRAPI_C_H */
