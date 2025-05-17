#ifndef TLDR_CPP_TLDR_H
#define TLDR_CPP_TLDR_H

#include <string>
#include <vector>
#include <cstdint>

namespace tldr_cpp_api {

// Structure to hold RAG query results
struct RagResult {
    std::string response;
    // Vector of (text, similarity_score, hash) tuples
    std::vector<std::tuple<std::string, float, uint64_t>> context_chunks;
};

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

#endif //TLDR_CPP_TLDR_H