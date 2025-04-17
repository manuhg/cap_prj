#ifndef TLDR_CPP_MAIN_H
#define TLDR_CPP_MAIN_H

#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include "db/database.h"
#include "db/postgres_database.h"
#include "db/sqlite_database.h"
#include "constants.h"

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
std::string extractTextFromPDF(const std::string &filename);
std::vector<std::string>
splitTextIntoChunks(const std::string &text, size_t max_chunk_size = 2000, size_t overlap = 20);
bool initializeDatabase();
int64_t saveEmbeddings(const std::vector<std::string> &chunks, const json &embeddings_response);
std::string sendEmbeddingsRequest(const json &request, const std::string& url);
json parseEmbeddingsResponse(const std::string &response_data);
int saveEmbeddingsThreadSafe(const std::vector<std::string> &batch, const json &embeddings_json);
void processChunkBatch(const std::vector<std::string> &batch, size_t batch_num, size_t total_batches, int &result_id);
void obtainEmbeddings(const std::vector<std::string> &chunks, size_t batch_size = 2, size_t num_threads = 2);

bool initializeSystem();
void cleanupSystem();
void addCorpus(const std::string &sourcePath);
void deleteCorpus(const std::string &corpusId);
void doRag(const std::string &conversationId);
void command_loop();
int main();
void queryRag(const std::string& user_query, const std::string& embeddings_url = EMBEDDINGS_URL, const std::string& chat_url = CHAT_URL);

#endif //TLDR_CPP_MAIN_H
