#pragma once

#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include "db/database.h"
#include "constants.h"

namespace tldr {

class CurlHandle {
    CURL *curl_;
    struct curl_slist *headers_;

public:
    CurlHandle();
    ~CurlHandle();

    CURL *get() { return curl_; }
    struct curl_slist *headers() { return headers_; }
    void setupEmbeddingsRequest(const std::string &json_str, std::string &response_data);
};

class TldrLib {
public:
    TldrLib();
    ~TldrLib();

    // Database operations
    void initializeDatabase(const std::string& dbPath);
    int64_t saveEmbeddings(const std::vector<std::string> &chunks, const json &embeddings_response);
    int saveEmbeddingsThreadSafe(const std::vector<std::string> &batch, const json &embeddings_json);

    // Document processing
    void processDocument(const std::string& filePath);
    std::string extractTextFromPDF(const std::string &filename);
    std::vector<std::string> splitTextIntoChunks(const std::string &text, size_t max_chunk_size, size_t overlap);
    void processChunkBatch(const std::vector<std::string_view> &batch, size_t batch_num, size_t total_batches, int &result_id);
    void obtainEmbeddings(const std::vector<std::string> &chunks, size_t batch_size, size_t num_threads);

    // RAG operations
    void doRag(const std::string &conversationId);
    void queryRag(const std::string& user_query, 
                 const std::string& embeddings_url = EMBEDDINGS_URL, 
                 const std::string& chat_url = CHAT_URL);
    std::string generateLLMResponse(const std::string& context, const std::string& user_query, const std::string& chat_url);

    // Helper methods
    std::string sendEmbeddingsRequest(const json &request, const std::string& url);
    json parseEmbeddingsResponse(const std::string &response_data);
    static std::string translatePath(const std::string& path);

private:
    std::unique_ptr<Database> db;
    static size_t WriteCallback(char *contents, size_t size, size_t nmemb, void *userp);
};

} // namespace tldr
