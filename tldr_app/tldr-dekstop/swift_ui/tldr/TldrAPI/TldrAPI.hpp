#ifndef TLDR_CPP_TLDR_H
#define TLDR_CPP_TLDR_H

#include <string>
#include "include/tldr_api.h"

// Forward declaration of C++ functions
int tldr_trial_main();

// C++ namespace for the TldrAPI
namespace TldrAPI {

/**
 * @brief Initialize the TLDR system
 * @return true if initialization was successful, false otherwise
 */
bool wrapper_initializeSystem();

/**
 * @brief Clean up the TLDR system
 */
void cleanupSystem();

/**
 * @brief Add a document to the corpus
 * @param sourcePath Path to the PDF file to add
 */
void addCorpus(const std::string& sourcePath);

/**
 * @brief Delete a document from the corpus
 * @param corpusId ID of the corpus to delete
 */
void deleteCorpus(const std::string& corpusId);

/**
 * @brief Query the RAG system
 * @param user_query The user's question
 * @param embeddings_url URL of the embeddings service (optional)
 * @param chat_url URL of the chat service (optional)
 */
void queryRag(const std::string& user_query,
              const std::string& embeddings_url = "http://localhost:8084/embeddings",
              const std::string& chat_url = "http://localhost:8088/v1/chat/completions");

} // namespace TldrAPI

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
