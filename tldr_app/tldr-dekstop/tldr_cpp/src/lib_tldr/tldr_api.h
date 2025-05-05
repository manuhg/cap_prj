#ifndef TLDR_CPP_TLDR_H
#define TLDR_CPP_TLDR_H

#include <string>

namespace tldr_cpp_api {

/**
 * @brief Initialize the TLDR system
 * @return true if initialization was successful, false otherwise
 */
bool initializeSystem();

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
 */
void queryRag(const std::string& user_query);

} // namespace tldr_cpp_api

// C API functions for Swift integration
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Test function for Swift to call into C++
 * @return An integer value (42) to confirm successful call
 */
int tldr_api_trial_tldr();

#ifdef __cplusplus
}
#endif

#endif // TLDR_CPP_TLDR_H