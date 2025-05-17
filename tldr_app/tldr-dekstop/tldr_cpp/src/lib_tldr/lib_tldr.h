#ifndef TLDR_CPP_MAIN_H
#define TLDR_CPP_MAIN_H

#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include "db/database.h"
#include "db/postgres_database.h"
#include "db/sqlite_database.h"
#include "constants.h"
#include <vector>
#include <string>
#include <cstdint>
#include <map>

// Directory name for storing vector cache files
constexpr const char* VECDUMP_DIR = "_vecdumps";
// #include "libs/sqlite_modern_cpp.h"
#include "vec_dump.h"
#include "npu_accelerator.h"

// Structure for similarity search results from the NPU accelerator
struct VectorSimilarityMatch {
    uint64_t hash;
    float score;
    std::string text; // Text associated with the hash (filled in by wrapper)
};

// Wrapper function for NPU similarity search
std::vector<std::tuple<std::string, float, uint64_t>> searchSimilarVectorsNPU(
    const std::vector<float>& query_vector,
    const std::string& corpus_dir,
    int k
);

// CURL wrapper class
class CurlHandle {
    CURL *curl_;
    struct curl_slist *headers_;

public:
    CurlHandle();
    ~CurlHandle();

    CURL *get() { return curl_; }
    struct curl_slist *headers() { return headers_; }

    // Configure CURL options for embeddings request
    void setupEmbeddingsRequest(const std::string &json_str, std::string &response_data);
};

// Function declarations
std::string translatePath(const std::string &path);
std::string extractTextFromPDF(const std::string &filename);
std::vector<std::string>
splitTextIntoChunks(const std::string &text, size_t max_chunk_size = 2000, size_t overlap = 20);
bool initializeDatabase();
int64_t saveEmbeddingsToDb(const std::vector<std::string_view> &chunks, const std::vector<std::vector<float>> &embeddings, const std::vector<uint64_t> &embeddings_hash = {});
std::string sendEmbeddingsRequest(const json &request, const std::string& url);
json parseEmbeddingsResponse(const std::string &response_data);
int saveEmbeddingsThreadSafe(const std::vector<std::string_view> &batch, const std::vector<std::vector<float>> &batch_embeddings, const std::vector<uint64_t> &embeddings_hash);

// Returns gathered embeddings and their hashes
std::pair<std::vector<std::vector<float>>, std::vector<uint64_t>> 
obtainEmbeddings(const std::vector<std::string> &chunks, const std::vector<std::string> &fileHashes, size_t batch_size = 2, size_t num_threads = 2);

// Process a single PDF file and add it to the corpus
void processPdfFile(const std::string& filePath, const std::string& fileHash);

// Function to add a file to the corpus
void addFileToCorpus(const std::string &sourcePath, const std::string &fileHash);

// Find all PDF files in a directory recursively
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
void addCorpus(const std::string &sourcePath);
void deleteCorpus(const std::string &corpusId);
// Structure to hold context chunk information
struct ContextChunk {
    std::string text;
    float similarity;
    uint64_t hash;
};

// Structure to hold RAG query results
struct RagResult {
    std::string response;
    std::vector<ContextChunk> context_chunks;
};

void doRag(const std::string &conversationId);
void command_loop();

/**
 * @brief Compute SHA-256 hash of one or more files using the shasum command-line utility
 * @param file_paths Vector of file paths to compute hashes for
 * @return std::map<std::string, std::string> Map of file paths to their corresponding SHA-256 hashes
 * @throws std::runtime_error If the shasum command fails or a file cannot be read
 */
std::map<std::string, std::string> computeFileHashes(const std::vector<std::string>& file_paths);

RagResult queryRag(const std::string& user_query, const std::string& corpus_dir = "/Users/manu/proj_tldr/corpus/current/");

#endif //TLDR_CPP_MAIN_H
