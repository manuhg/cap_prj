#ifndef TLDR_CPP_TLDR_H
#define TLDR_CPP_TLDR_H

#include <string>

namespace TldrAPI {

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
 * @param embeddings_url URL of the embeddings service (optional)
 * @param chat_url URL of the chat service (optional)
 */
void queryRag(const std::string& user_query, 
              const std::string& embeddings_url = "http://localhost:8084/embeddings",
              const std::string& chat_url = "http://localhost:8088/v1/chat/completions");

} // namespace tldr

#endif // TLDR_CPP_TLDR_H 
