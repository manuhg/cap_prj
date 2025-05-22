#ifndef TLDR_API_C_H
#define TLDR_API_C_H

#include <stddef.h>  // for size_t
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char* text;
    char* file_path;
    char* file_name;
    char* title;
    char* author;
    int page_count;
    int page_number;
    float similarity;
    unsigned long long hash;
} CtxChunkMetaC;

typedef struct {
    char* response;
    CtxChunkMetaC* context_chunks;
    size_t context_chunks_count;
    int referenced_document_count;
} RagResultC;

// Initialize the TLDR system
bool tldr_initializeSystem(void);

// Clean up the TLDR system
void tldr_cleanupSystem(void);

// Add a document to the corpus
void tldr_addCorpus(const char* sourcePath);

// Delete a document from the corpus
void tldr_deleteCorpus(const char* corpusId);

// Query the RAG system
RagResultC* tldr_queryRag(const char* user_query, const char* corpus_dir);

// Free a RagResult
void tldr_freeRagResult(RagResultC* result);

void tldr_freeString(char* str);
#ifdef __cplusplus
}
#endif

#endif // TLDR_API_C_H
