#ifndef TLDR_CPP_TLDR_H
#define TLDR_CPP_TLDR_H

#include <string>
#include <vector>
#include "definitions.h"

namespace tldr_cpp_api {


/**
 * @brief Initialize the TLDR system
 * @param chat_model_path Path to the chat model file (leave empty to use default)
 * @param embeddings_model_path Path to the embeddings model file (leave empty to use default)
 * @return true if initialization was successful, false otherwise
 */
bool initializeSystem(const std::string& chat_model_path, const std::string& embeddings_model_path);

/**
 * @brief Clean up the TLDR system
 */
void cleanupSystem();

/**
 * @brief Add a document or directory of documents to the corpus
 * @param sourcePath Path to the PDF file or directory containing PDFs to add
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
 * @param corpus_dir Directory containing the corpus (defaults to current corpus)
 * @return RagResult containing the response and context chunks
 */
RagResult queryRag(const std::string& user_query, const std::string& corpus_dir = "/Users/manu/proj_tldr/corpus/current/");

/**
 * @brief Format the RAG result and its context metadata into a single string
 * @param result The RagResult object containing the LLM response and context chunks
 * @return A formatted string containing the response and all context with metadata
 */
std::string printRagResult(const RagResult& result);

} // namespace tldr_cpp_api

// C API functions for Swift integration
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Test function for Swift to call into C++
 * @return An integer value (42) to confirm successful call
 */
// int tldr_api_trial_tldr();

#ifdef __cplusplus
}
#endif

#endif //TLDR_CPP_TLDR_H