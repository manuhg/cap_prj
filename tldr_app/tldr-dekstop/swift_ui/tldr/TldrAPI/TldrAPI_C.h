#ifndef TLDRAPI_C_H
#define TLDRAPI_C_H

#ifdef __cplusplus
extern "C" {
#endif

// Test function that can be called from Swift
int tldr_api_trial_tldr(void);

// Returns 1 if initialization was successful, 0 otherwise
bool tldr_initializeSystem(void);

// Cleans up the system
void tldr_cleanupSystem(void);

// Adds a corpus from a PDF file path
void tldr_addCorpus(const char* sourcePath);

// Deletes a corpus by ID
void tldr_deleteCorpus(const char* corpusId);

// Queries the RAG system
void tldr_queryRag(const char* user_query, const char* embeddings_url, const char* chat_url);

#ifdef __cplusplus
}
#endif

#endif /* TLDRAPI_C_H */
