//
//

#ifndef TLDR_CPP_MAIN_H
#define TLDR_CPP_MAIN_H

#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include "db/database.h"
#include "db/sqlite_database.h"
#include "db/postgres_database.h"

// Text processing constants
#define PAGE_DELIMITER "\n\n"
#define AVG_WORDS_PER_SENTENCE 6
#define AVG_CHARS_PER_WORD 5
#define CHUNK_N_SENTENCES 10
#define CHUNK_N_OVERLAP 20 // overlap in characters at start and end
#define CHUNK_N_CHARS ((CHUNK_N_SENTENCES * AVG_WORDS_PER_SENTENCE * AVG_CHARS_PER_WORD)+(CHUNK_N_OVERLAP*2))
#define MAX_CHARS_PER_BATCH 2048

// Database constants
#define USE_POSTGRES false  // Set to true to use PostgreSQL, false for SQLite
#define DB_PATH "datastore/embeddings.db"
#define PG_CONNECTION "dbname=tldr_embeddings user=postgres password=postgres host=localhost port=5432"

// HTTP request constants
#define EMBEDDINGS_URL "http://localhost:8080/embedding"
#define CONNECT_TIMEOUT_SECONDS 5
#define REQUEST_TIMEOUT_SECONDS 30
#define MAX_RETRIES 3
#define RETRY_DELAY_MS 1000

// CURL wrapper class
class CurlHandle {
    CURL* curl_;
    struct curl_slist* headers_;

public:
    CurlHandle();
    ~CurlHandle();
    
    CURL* get() { return curl_; }
    struct curl_slist* headers() { return headers_; }
    
    // Configure CURL options for embeddings request
    void setupEmbeddingsRequest(const std::string& json_str, std::string& response_data);
};

// Function declarations
void test_extractTextFromPDF();
std::string extractTextFromPDF(const std::string &filename);
std::vector<std::string> splitTextIntoChunks(const std::string &text, size_t max_chunk_size = 2000, size_t overlap = 20);
bool initializeDatabase();
int64_t saveEmbeddings(const std::vector<std::string> &chunks, const json &embeddings_response);
std::string sendEmbeddingsRequest(const json& request);
json parseEmbeddingsResponse(const std::string& response_data);
void saveEmbeddingsThreadSafe(const std::vector<std::string>& batch, const json& embeddings_json);
void processChunkBatch(const std::vector<std::string>& batch, size_t batch_num, size_t total_batches);
bool initializeSystem();
void cleanupSystem();
void obtainEmbeddings(const std::vector<std::string> &chunks, size_t batch_size = 2, size_t num_threads = 2);
void addCorpus(const std::string &sourcePath);
void deleteCorpus(const std::string &corpusId);
void doRag(const std::string &conversationId);
void command_loop();
int main();

struct embeddings_request {
    std::vector<std::string> input;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(embeddings_request, input)
#endif //TLDR_CPP_MAIN_H
