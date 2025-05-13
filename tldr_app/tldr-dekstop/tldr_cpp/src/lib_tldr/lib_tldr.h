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
void processChunkBatch(const std::vector<std::string> &batch, size_t batch_num, size_t total_batches, int &result_id);
// Returns gathered embeddings and their hashes
std::pair<std::vector<std::vector<float>>, std::vector<uint64_t>> 
obtainEmbeddings(const std::vector<std::string> &chunks, size_t batch_size = 2, size_t num_threads = 2);

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
void doRag(const std::string &conversationId);
void command_loop();
void queryRag(const std::string& user_query, const std::string& corpus_dir = "/Users/manu/proj_tldr/corpus/current/");

#endif //TLDR_CPP_MAIN_H
