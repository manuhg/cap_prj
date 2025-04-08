#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <pqxx/pqxx>
#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-page.h>
#include <thread>
#include <mutex>

#include "main.h"
#include <regex>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <memory>

using json = nlohmann::json;

// Global database instance
std::unique_ptr<tldr::Database> g_db;

// Global mutex for thread synchronization
std::mutex g_mutex;

void test_extractTextFromPDF() {
    std::string testFile = "../../corpus/current/97-things-every-software-architect-should-know.pdf";

    std::string extractedText = extractTextFromPDF(testFile);
    std::cout << extractedText.length() << std::endl;

    assert(extractedText.length()==268979);
    std::cout << "test_extractTextFromPDF passed!" << std::endl;
    std::cout << "Extracted text (first 50 chars): \n=========" << extractedText.substr(0, 50) << "\n=========" <<
            std::endl;
}

//////
// Simple function to extract text from a PDF file
std::string extractTextFromPDF(const std::string &filename) {
    // Load the PDF document
    poppler::document *doc = poppler::document::load_from_file(filename);

    // Check if the document loaded successfully
    if (!doc) {
        std::cerr << "Error opening PDF file." << std::endl;
        return "";
    }

    std::string text;

    // Iterate through all pages
    for (int i = 0; i < doc->pages(); ++i) {
        // Get the current page
        poppler::page *page = doc->create_page(i);
        if (page) {
            // Extract text from the page and sanitize UTF-8
            poppler::byte_array utf8_data = page->text().to_utf8();
            std::string page_text;
            // Only keep ASCII characters for now
            for (unsigned char c: utf8_data) {
                if (c < 128) {
                    // ASCII range
                    page_text += c;
                }
            }

            text.append(page_text + PAGE_DELIMITER);
            delete page;
        }
    }

    // Clean up
    delete doc;

    return text;
}

// function to split text into fixed-size chunks with overlap
std::vector<std::string> splitTextIntoChunks(const std::string &text, size_t max_chunk_size, size_t overlap) {
    std::vector<std::string> chunks;
    size_t pos = 0;
    const size_t text_len = text.length();

    while (pos < text_len) {
        // Calculate end position for this chunk
        size_t chunk_end = std::min(pos + max_chunk_size, text_len);

        // Add the chunk
        int num_chars = chunk_end - pos;
        chunks.push_back(text.substr(pos, num_chars));

        // Move position for next chunk, accounting for overlap
        pos = num_chars > overlap ? chunk_end - overlap : chunk_end;
    }

    return chunks;
}




bool initializeDatabase() {
    if (!g_db) {
        if (USE_POSTGRES) {
            g_db = std::make_unique<tldr::PostgresDatabase>(PG_CONNECTION);
        } else {
            g_db = std::make_unique<tldr::SQLiteDatabase>(DB_PATH);
        }

        if (!g_db->initialize()) {
            std::cerr << "Failed to initialize database" << std::endl;
            return false;
        }
    }
    return true;
}

int64_t saveEmbeddings(const std::vector<std::string> &chunks, const json &embeddings_response) {
    if (!g_db) {
        std::cerr << "Database not initialized" << std::endl;
        return -1;
    }

    return g_db->saveEmbeddings(chunks, embeddings_response);
}

size_t WriteCallback(char *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string *) userp)->append((char *) contents, size * nmemb);
    return size * nmemb;
}

// CURL wrapper class implementation
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
    curl_easy_setopt(curl_, CURLOPT_URL, EMBEDDINGS_URL);
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, json_str.c_str());
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers_);
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT, CONNECT_TIMEOUT_SECONDS);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, REQUEST_TIMEOUT_SECONDS);
}

// Send HTTP request to embeddings service
std::string sendEmbeddingsRequest(const json &request) {
    CurlHandle curl;
    std::string json_str = request.dump();
    std::string response_data;

    curl.setupEmbeddingsRequest(json_str, response_data);

    CURLcode res = curl_easy_perform(curl.get());
    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("CURL request failed: ") + curl_easy_strerror(res));
    }

    return response_data;
}

// Parse response and extract embeddings
json parseEmbeddingsResponse(const std::string &response_data) {
    try {
        json response = json::parse(response_data);
        if (!response.contains("embedding")) {
            // Debug print the full response
            std::cerr << "Full response: " << response.dump(2) << std::endl;
            throw std::runtime_error("Invalid response format: missing 'embedding' field");
        }
        return response;
    } catch (const json::parse_error &e) {
        std::cerr << "Raw response data: " << response_data << std::endl;
        throw std::runtime_error(std::string("Failed to parse response: ") + e.what());
    }
}

// Save embeddings to database with thread safety
void saveEmbeddingsThreadSafe(const std::vector<std::string> &batch, const json &embeddings_json) {
    std::lock_guard<std::mutex> lock(g_mutex);
    int64_t saved_id = saveEmbeddings(batch, embeddings_json);
    if (saved_id < 0) {
        throw std::runtime_error("Failed to save embeddings to database");
    }
}

// Main function to process a batch of chunks
void processChunkBatch(const std::vector<std::string> &batch, size_t batch_num, size_t total_batches) {
    // Process each text chunk individually since llama.cpp server expects single text input
    std::vector<json> embeddings_list;
    embeddings_list.reserve(batch.size());

    for (int retry = 0; retry < MAX_RETRIES; retry++) {
        if (retry > 0) {
            std::cout << "Retrying batch " << batch_num + 1 << " (attempt " << retry + 1
                    << " of " << MAX_RETRIES << ")" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
        }

        std::cout << "Processing batch " << batch_num + 1 << " of " << total_batches << std::endl;

        try {
            embeddings_list.clear();

            // Prepare request for llama.cpp server
            json request_payload = {
                {"prompt", batch[0]},  // Use first chunk as prompt
                {"n_predict", 0},     // No text generation
                {"stream", false},    // No streaming
                {"embedding", true}   // Request embeddings
            };

            std::string response_data = sendEmbeddingsRequest(request_payload);
            json embeddings_json = parseEmbeddingsResponse(response_data);
            saveEmbeddingsThreadSafe(batch, embeddings_json);
            return; // Success
        } catch (const std::exception &e) {
            if (retry == MAX_RETRIES - 1) {
                throw std::runtime_error("Failed after " + std::to_string(MAX_RETRIES) +
                                         " attempts: " + std::string(e.what()));
            }
            std::cerr << "Attempt " << retry + 1 << " failed: " << e.what() << std::endl;
        }
    }

    throw std::runtime_error("Failed to process batch after " + std::to_string(MAX_RETRIES) + " attempts");
}

bool initializeSystem() {
    // Initialize database
    if (!initializeDatabase()) {
        std::cerr << "Failed to initialize database" << std::endl;
        return false;
    }

    // Initialize CURL globally
    CURLcode curl_init = curl_global_init(CURL_GLOBAL_DEFAULT); //question: is this efficient or should we use a thread-specific instance? or use protobuf
    if (curl_init != CURLE_OK) {
        std::cerr << "Failed to initialize CURL: " << curl_easy_strerror(curl_init) << std::endl;
        return false;
    }

    return true;
}

void cleanupSystem() {
    // Cleanup CURL
    curl_global_cleanup();

    // Cleanup database
    g_db.reset();
}


void obtainEmbeddings(const std::vector<std::string> &chunks, size_t batch_size, size_t num_threads) {
    // System initialization is now handled by initializeSystem()
    const size_t total_batches = (chunks.size() + batch_size - 1) / batch_size;
    std::cout << "Processing " << chunks.size() << " chunks in " << total_batches
            << " batches using " << num_threads << " threads\n";

    try {
        for (size_t i = 0; i < chunks.size(); i += batch_size * num_threads) {
            std::vector<std::thread> threads;
            threads.reserve(num_threads);

            // Launch threads for each batch in this group
            for (size_t t = 0; t < num_threads && (i + t * batch_size) < chunks.size(); ++t) {
                size_t start = i + t * batch_size;
                size_t end = std::min(start + batch_size, chunks.size());

                std::vector<std::string> batch(chunks.begin() + start, chunks.begin() + end);
                threads.emplace_back(processChunkBatch, std::ref(batch), start / batch_size, total_batches);
            }

            // Wait for all threads in this group to complete
            for (auto &thread: threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Error processing chunks: " << e.what() << std::endl;
        throw; // Re-throw to allow proper cleanup
    }
}

void addCorpus(const std::string &sourcePath) {
    std::string doc_text = extractTextFromPDF(sourcePath);
    if (doc_text.empty()) {
        std::cerr << "Error: No text extracted from PDF." << std::endl;
        return;
    }
    // print text length only
    std::cout << "Extracted text length: " << doc_text.length() << std::endl;
    const std::vector<std::string> chunks = splitTextIntoChunks(doc_text, 500, 20);
    std::cout << "Number of chunks: " << chunks.size() << std::endl;
    obtainEmbeddings(chunks);
    //TODO save the json to database and return the id
}

void deleteCorpus(const std::string &corpusId) {
    // Implement the function
    std::cout << "DELETE_CORPUS action with corpus_id: " << corpusId << std::endl;
}

void doRag(const std::string &conversationId) {
    // Implement the function
    std::cout << "DO_RAG action with conversation_id: " << conversationId << std::endl;

    try {
        pqxx::connection C("dbname=testdb user=postgres password=secret hostaddr=127.0.0.1 port=5432");
        pqxx::work W(C);

        std::string query =
                "SELECT * FROM conversations WHERE id = " + W.quote(conversationId) + " ORDER BY created_at";
        pqxx::result R = W.exec(query);

        for (auto row: R) {
            std::cout << "ID: " << row["id"].c_str() << ", Created At: " << row["created_at"].c_str() << std::endl;
        }

        W.commit();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}

void command_loop() {
    std::string input;
    std::map<std::string, std::function<void(const std::string &)> > actions = {
        {"do-rag", doRag},
        {"add-corpus", addCorpus},
        {"delete-corpus", deleteCorpus}
    };

    while (true) {
        std::cout << "Enter command: ";
        std::getline(std::cin, input);
        std::istringstream iss(input);
        std::string command, argument;
        iss >> command >> argument;

        auto it = actions.find(command);
        if (it != actions.end()) {
            it->second(argument);
        } else if (command == "exit") {
            break;
        } else {
            std::cout << "Unknown command: " << command << std::endl;
        }
    }
}


int main() {
    // Initialize system
    if (!initializeSystem()) {
        std::cerr << "Failed to initialize system" << std::endl;
        return 1;
    }

    std::string testFile = "../corpus/current/97-things-every-software-architect-should-know.pdf";

    std::cout << "Testing addCorpus with file: " << testFile << std::endl;
    addCorpus(testFile);

    // Cleanup system
    cleanupSystem();

    return 0;
}
