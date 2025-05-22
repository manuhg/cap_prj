#ifndef TLDR_CPP_MAIN_H
#define TLDR_CPP_MAIN_H
#include "definitions.h"
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include "db/database.h"
#include "db/postgres_database.h"
#include "db/sqlite_database.h"
#include <libpq-fe.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <sstream>

// Constants
const std::string PAGE_DELIMITER = "\n--- PAGE BREAK ---\n";

// Helper function to extract content from XML tags
std::string extract_xml_content(const std::string& xml);

// Collect PDF files from a path (file or directory)
// Returns a vector of PDF file paths, or an empty vector if no files found or on error
std::vector<std::string> collectPdfFiles(const std::string& path);

// #include "libs/sqlite_modern_cpp.h"
#include "vec_dump.h"
#include "npu_accelerator.h"


// Function declarations
std::string translatePath(const std::string &path);

// Extract metadata from a PDF file
PdfMetadata getPdfMetadata(const std::string &filename);

// Extract document data including metadata and page texts from a PDF file
DocumentData extractDocumentDataFromPDF(const std::string &filename);

// Kept for backward compatibility
[[deprecated("Use extractDocumentDataFromPDF instead")]]
std::string extractTextFromPDF(const std::string &filename);
// Get page boundaries (end positions) for a document
std::vector<size_t> getPageBoundaries(const DocumentData& docData);

// Split document text into chunks with page tracking
void splitTextIntoChunks(DocumentData& docData, size_t max_chunk_size = 2000, size_t overlap = 20);
// Database connection management
bool initializeDatabase(const std::string& conninfo = "");
void closeDatabase();

// Save or update document metadata in the database
bool saveOrUpdateDocumentInDB(const std::string& fileHash,
                         const std::string& filePath,
                         const DocumentData& docData);

// Save embeddings to the database with page numbers and file hash reference
int64_t saveEmbeddingsToDb(const std::vector<std::string_view> &chunks, 
                          const std::vector<std::vector<float>> &embeddings, 
                          const std::vector<uint64_t> &embeddings_hash,
                          const std::vector<int>& chunkPageNums,
                          const std::string& fileHash);
std::string sendEmbeddingsRequest(const json &request, const std::string& url);
json parseEmbeddingsResponse(const std::string &response_data);
int saveEmbeddingsThreadSafe(const std::vector<std::string_view> &batch,
    const std::vector<std::vector<float>> &batch_embeddings, 
    const std::vector<uint64_t> &embeddings_hash,
    const std::vector<int>& chunkPageNums,
    const std::string &fileHash);

// Returns gathered embeddings and their hashes
std::pair<std::vector<std::vector<float>>, std::vector<uint64_t>> 
obtainEmbeddings(const std::vector<std::string> &chunks, 
                const std::vector<int>& chunkPageNums,
                const std::string &fileHash,
                size_t batch_size, size_t num_threads);

// Delete all embeddings for a specific file hash
bool deleteFileEmbeddingsFromDB(const std::string& fileHash);

// Function to add a file to the corpus
bool addFileToCorpus(const std::string &sourcePath, const std::string &fileHash);

// Find all PDF files in a directory recursively
// Generic function to find files of a specific type recursively
void findFilesOfTypeRecursively(const std::filesystem::path& path, std::vector<std::string>& files, const std::string& extension);

// Deprecated: Use findFilesOfTypeRecursively instead
void findPdfFiles(const std::filesystem::path& path, std::vector<std::string>& pdfFiles);

// Include vector dump functionality
#include "vec_dump.h"
std::map<uint64_t, float> npuCosineSimSearchWrapper(
    const float *queryVector, const int queryVectorDimensions, const int32_t k = 5,
    const char *corpusDir = "/Users/manu/proj_tldr/corpus/current/",
    const char *modelPath =
            "/Users/manu/proj_tldr/tldr-dekstop/release-products/artefacts/CosineSimilarityBatched.mlmodelc");

bool initializeSystem();
void cleanupSystem();
WorkResult addCorpus(const std::string &sourcePath);
void deleteCorpus(const std::string &corpusId);
// Structure to hold context chunk information


void doRag(const std::string &conversationId);
void command_loop();

/**
 * @brief Format the RAG result and its context metadata into a single string
 * @param result The RagResult object containing the LLM response and context chunks
 * @return A formatted string containing the response and all context with metadata
 */
std::string printRagResult(const RagResult &result);

/**
 * @brief Compute SHA-256 hash of one or more files using the shasum command-line utility
 * @param file_paths Vector of file paths to compute hashes for
 * @return std::map<std::string, std::string> Map of file paths to their corresponding SHA-256 hashes
 * @throws std::runtime_error If the shasum command fails or a file cannot be read
 */
bool computeFileHashes(const std::vector<std::string>& file_paths, std::map<std::string, std::string> &file_hashes, WorkResult &result);

RagResult queryRag(const std::string& user_query, const std::string& corpus_dir = "/Users/manu/proj_tldr/corpus/current/");

#endif //TLDR_CPP_MAIN_H
