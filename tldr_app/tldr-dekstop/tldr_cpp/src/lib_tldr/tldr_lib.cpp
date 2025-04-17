#include "tldr_lib.h"
#include "db/sqlite_database.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <poppler-document.h>
#include <poppler-page.h>
#include <cstdlib>

namespace tldr {

CurlHandle::CurlHandle() : curl_(curl_easy_init()), headers_(nullptr) {
    if (!curl_) {
        throw std::runtime_error("Failed to initialize CURL");
    }
    headers_ = curl_slist_append(headers_, "Content-Type: application/json");
}

CurlHandle::~CurlHandle() {
    if (headers_) curl_slist_free_all(headers_);
    if (curl_) curl_easy_cleanup(curl_);
}

void CurlHandle::setupEmbeddingsRequest(const std::string &json_str, std::string &response_data) {
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, json_str.c_str());
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers_);
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, TldrLib::WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 30L);
}

TldrLib::TldrLib() {}

TldrLib::~TldrLib() = default;

void TldrLib::initializeDatabase(const std::string& dbPath) {
    db = std::make_unique<SQLiteDatabase>(dbPath);
    db->initialize();
}

void TldrLib::processDocument(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();

    // Extract text from PDF if it's a PDF file
    std::string processedText = text;
    if (text.substr(0, 4) == "%PDF") {
        // Create a temporary file to store the PDF content
        std::string tempFile = "/tmp/temp.pdf";
        std::ofstream out(tempFile, std::ios::binary);
        out.write(text.c_str(), text.size());
        out.close();

        processedText = extractTextFromPDF(tempFile);
        std::remove(tempFile.c_str());
    }

    // Split text into chunks
    auto chunks = splitTextIntoChunks(processedText, MAX_CHUNK_SIZE, CHUNK_N_OVERLAP);
    
    // Get embeddings and save to database
    obtainEmbeddings(chunks, BATCH_SIZE, NUM_THREADS);
}

std::string TldrLib::extractTextFromPDF(const std::string &filename) {
    poppler::document *doc = poppler::document::load_from_file(filename);

    if (!doc) {
        std::cerr << "Error opening PDF file." << std::endl;
        return "";
    }

    std::string text;

    for (int i = 0; i < doc->pages(); ++i) {
        poppler::page *page = doc->create_page(i);
        std::string page_text;
        page_text.reserve(1024 * 4);
        if (page) {
            page_text.clear();
            poppler::byte_array utf8_data = page->text().to_utf8();

            for (unsigned char c: utf8_data) {
                if (c < 128) {
                    page_text += c;
                }
            }

            text.append(page_text + PAGE_DELIMITER);
            delete page;
        }
    }

    delete doc;
    return text;
}

std::vector<std::string> TldrLib::splitTextIntoChunks(const std::string &text, size_t max_chunk_size, size_t overlap) {
    std::vector<std::string> chunks;
    size_t pos = 0;
    const size_t text_len = text.length();

    while (pos < text_len) {
        size_t chunk_end = std::min(pos + max_chunk_size, text_len);
        int num_chars = chunk_end - pos;
        chunks.push_back(text.substr(pos, num_chars));
        pos = num_chars > overlap ? chunk_end - overlap : chunk_end;
    }

    return chunks;
}

int64_t TldrLib::saveEmbeddings(const std::vector<std::string> &chunks, const json &embeddings_response) {
    if (!db) {
        std::cerr << "Database not initialized" << std::endl;
        return -1;
    }
    return db->saveEmbeddings(chunks, embeddings_response);
}

size_t TldrLib::WriteCallback(char *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string *) userp)->append((char *) contents, size * nmemb);
    return size * nmemb;
}

std::string TldrLib::sendEmbeddingsRequest(const json &request, const std::string& url) {
    CurlHandle curl;
    std::string json_str = request.dump();
    std::string response_data;

    curl.setupEmbeddingsRequest(json_str, response_data);
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());

    CURLcode res = curl_easy_perform(curl.get());
    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("CURL request failed: ") + curl_easy_strerror(res));
    }

    return response_data;
}

json TldrLib::parseEmbeddingsResponse(const std::string &response_data) {
    try {
        json response = json::parse(response_data);

        if (!response.contains("data") || !response["data"].is_array()) {
            std::cerr << "Full response: " << response.dump(2) << std::endl;
            throw std::runtime_error("Invalid response format: missing 'data' array");
        }

        json embeddings_json;
        embeddings_json["embeddings"] = json::array();

        for (const auto &item: response["data"]) {
            if (item.contains("embedding") && item["embedding"].is_array()) {
                embeddings_json["embeddings"].push_back(item["embedding"]);
            }
        }

        if (embeddings_json["embeddings"].empty()) {
            std::cerr << "No embeddings found in the response" << std::endl;
            throw std::runtime_error("No embeddings found in the response");
        }

        return embeddings_json;
    } catch (const json::parse_error &e) {
        std::cerr << "Raw response data: " << response_data << std::endl;
        throw std::runtime_error(std::string("Failed to parse response: ") + e.what());
    }
}

int TldrLib::saveEmbeddingsThreadSafe(const std::vector<std::string> &batch, const json &embeddings_json) {
    int64_t saved_id = saveEmbeddings(batch, embeddings_json);
    if (saved_id < 0) {
        throw std::runtime_error("Failed to save embeddings to database");
    }
    return saved_id;
}

void TldrLib::processChunkBatch(const std::vector<std::string_view> &batch, size_t batch_num, size_t total_batches, int &result_id) {
    std::vector<json> embeddings_list;
    embeddings_list.reserve(batch.size());

    int total_length = 0;
    for (const auto &chunk: batch) {
        total_length += chunk.length();
    }
    result_id = -1;

    if (batch.empty() || total_length <= 0) {
        std::cerr << "Error: Empty Batch!" << std::endl;
        return;
    }
    std::cout << "Received text length: " << total_length << "; batch size:" << batch.size() << std::endl;

    for (int retry = 0; retry < MAX_RETRIES; retry++) {
        if (retry > 0) {
            std::cout << "Retrying batch " << batch_num + 1 << " (attempt " << retry + 1
                    << " of " << MAX_RETRIES << ")" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
        }

        std::cout << "Processing batch " << batch_num + 1 << " of " << total_batches << std::endl;

        try {
            embeddings_list.clear();
            json request_payload = {
                {"input", std::vector<std::string>(batch.begin(), batch.end())}
            };

            std::string response = sendEmbeddingsRequest(request_payload, EMBEDDINGS_URL);
            json embeddings_json = parseEmbeddingsResponse(response);
            result_id = saveEmbeddingsThreadSafe(std::vector<std::string>(batch.begin(), batch.end()), embeddings_json);
            return;
        } catch (const std::exception &e) {
            std::cerr << "Error processing batch " << batch_num + 1 << ": " << e.what() << std::endl;
            if (retry == MAX_RETRIES - 1) {
                throw;
            }
        }
    }
}

void TldrLib::obtainEmbeddings(const std::vector<std::string> &chunks, size_t batch_size, size_t num_threads) {
    if (chunks.empty()) {
        return;
    }

    std::vector<std::thread> threads;
    std::vector<int> result_ids(chunks.size() / batch_size + 1, -1);
    threads.reserve(num_threads);

    for (size_t i = 0; i < chunks.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, chunks.size());
        std::vector<std::string_view> batch(chunks.begin() + i, chunks.begin() + end);
        size_t batch_num = i / batch_size;

        while (threads.size() >= num_threads) {
            for (auto it = threads.begin(); it != threads.end();) {
                if (it->joinable()) {
                    it->join();
                    it = threads.erase(it);
                } else {
                    ++it;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        threads.emplace_back([this, batch, batch_num, total_batches = (chunks.size() + batch_size - 1) / batch_size, &result_ids]() {
            processChunkBatch(batch, batch_num, total_batches, result_ids[batch_num]);
        });
    }

    for (auto &thread: threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

std::string TldrLib::generateLLMResponse(const std::string& context, const std::string& user_query, const std::string& chat_url) {
    CurlHandle curl;
    std::string response_data;

    json request = {
        {"messages", {
            {"role", "system"},
            {"content", "You are a helpful assistant that answers questions based on the provided context. "
                       "If the answer cannot be found in the context, say 'I cannot answer this question based on the provided context.'"}
        }},
        {"context", context},
        {"query", user_query}
    };

    curl.setupEmbeddingsRequest(request.dump(), response_data);
    curl_easy_setopt(curl.get(), CURLOPT_URL, chat_url.c_str());

    CURLcode res = curl_easy_perform(curl.get());
    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("CURL request failed: ") + curl_easy_strerror(res));
    }

    try {
        json response = json::parse(response_data);
        if (response.contains("response")) {
            return response["response"].get<std::string>();
        }
        throw std::runtime_error("Invalid response format from LLM service");
    } catch (const json::parse_error& e) {
        throw std::runtime_error(std::string("Failed to parse LLM response: ") + e.what());
    }
}

void TldrLib::doRag(const std::string &conversationId) {
    // Implementation remains the same
}

void TldrLib::queryRag(const std::string& user_query, const std::string& embeddings_url, const std::string& chat_url) {
    // Implementation remains the same
}

std::string TldrLib::translatePath(const std::string& path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }
    const char* home = std::getenv("HOME");
    if (!home) {
        return path;
    }
    return std::string(home) + path.substr(1);
}

} // namespace tldr
