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
#include <cstdlib>
#include <regex>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <memory>

#include "lib_tldr.h"
#include "llama.h"
#include "npu_accelerator.h"
#include "lib_tldr/llm-wrapper.h"
#include "lib_tldr/constants.h"

using json = nlohmann::json;

// Global database instance
std::unique_ptr<tldr::Database> g_db;
// Global LLM Manager instance
tldr::LlmManager g_llm_manager;

#if !USE_POSTGRES
// Global mutex for thread synchronization
// std::mutex g_mutex;
#endif

std::string translatePath(const std::string &path) {
    std::string result = path;

    // Expand tilde (~)
    if (!result.empty() && result[0] == '~') {
        const char *home = std::getenv("HOME");
        if (home) {
            result.replace(0, 1, home);
        }
    }

    // Expand $VARS
    static const std::regex env_pattern(R"(\$([A-Za-z_]\w*))");
    std::string expanded;
    std::sregex_iterator it(result.begin(), result.end(), env_pattern);
    std::sregex_iterator end;

    size_t last_pos = 0;
    for (; it != end; ++it) {
        const std::smatch &match = *it;
        expanded.append(result, last_pos, match.position() - last_pos);
        const char *val = std::getenv(match[1].str().c_str());
        expanded.append(val ? val : "");
        last_pos = match.position() + match.length();
    }
    expanded.append(result, last_pos);

    return expanded;
}

int calc_batch_chars(const std::vector<std::string_view> &batch) {
    int total_length = 0;
    if (batch.empty()) {
        return 0;
    }
    for (const auto &c: batch) {
        total_length += c.length();
    }
    return total_length;
}

// Simple function to extract text from a PDF file
std::string extractTextFromPDF(const std::string &filename) {
    // Load the PDF document
    std::string expanded_path = translatePath(filename);
    poppler::document *doc = poppler::document::load_from_file(expanded_path);

    // Check if the document loaded successfully
    if (!doc) {
        std::cerr << "Error opening PDF file at path: " << expanded_path << std::endl;
        return "";
    }

    std::string text;

    // Iterate through all pages
    for (int i = 0; i < doc->pages(); ++i) {
        // Get the current page
        poppler::page *page = doc->create_page(i);
        std::string page_text;
        page_text.reserve(1024 * 4);
        if (page) {
            page_text.clear();
            // Extract text from the page and sanitize UTF-8
            poppler::byte_array utf8_data = page->text().to_utf8();

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

    // Clean up the document
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
    std::cout << "Initialize the database" << std::endl;
    if (!g_db) {
        if (USE_POSTGRES) {
            g_db = std::make_unique<tldr::PostgresDatabase>(PG_CONNECTION);
        } else {
            g_db = std::make_unique<tldr::SQLiteDatabase>(translatePath(DB_PATH));
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
    //TODO add chunk and embedding hash ids via a db function

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
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, json_str.c_str());
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers_);
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT, CONNECT_TIMEOUT_SECONDS);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, REQUEST_TIMEOUT_SECONDS);
}

// Send HTTP request to embeddings service
std::string sendEmbeddingsRequest(const json &request, const std::string& url) {
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

// Parse response and extract embeddings
json parseEmbeddingsResponse(const std::string &response_data) {
    try {
        json response = json::parse(response_data);

        // Validate the response structure
        if (!response.contains("data") || !response["data"].is_array()) {
            std::cerr << "Full response: " << response.dump(2) << std::endl;
            throw std::runtime_error("Invalid response format: missing 'data' array");
        }

        // Create a new JSON object to store embeddings
        json embeddings_json;
        embeddings_json["embeddings"] = json::array();

        // Extract embeddings from each data item
        for (const auto &item: response["data"]) {
            if (item.contains("embedding") && item["embedding"].is_array()) {
                embeddings_json["embeddings"].push_back(item["embedding"]);
            }
        }

        // Verify we extracted embeddings
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

// Save embeddings to database with thread safety
int saveEmbeddingsThreadSafe(const std::vector<std::string> &batch, const std::vector<std::vector<float>> &batch_embeddings) {
    json embeddings_json;
    embeddings_json["embeddings"] = json::array();
    for(const auto& emb : batch_embeddings) {
        embeddings_json["embeddings"].push_back(emb);
    }

    if (embeddings_json["embeddings"].empty()) {
        std::cerr << "  No embeddings generated for this batch." << std::endl;
        return -1;
    }
#if !USE_POSTGRES
    // std::lock_guard<std::mutex> lock(g_mutex);
#endif
    int64_t saved_id = saveEmbeddings(batch, embeddings_json);
    if (saved_id < 0) {
        throw std::runtime_error("Failed to save embeddings to database");
    }
    return saved_id;
}

void processChunkBatch(const std::vector<std::string_view> &batch, size_t batch_num, size_t total_batches,
                       int &result_id) {
    // Process each text chunk individually since llama.cpp server expects single text input
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
            std::vector<std::vector<float>> batch_embeddings = g_llm_manager.get_embeddings(batch);
            if (batch_embeddings.size() != batch.size()) {
                std::cerr << "  Warning: Mismatch between input chunks (" << batch.size()
                          << ") and generated embeddings (" << batch_embeddings.size()
                          << ") for this batch. " << std::endl;
            }


            result_id = saveEmbeddingsThreadSafe(std::vector<std::string>(batch.begin(), batch.end()), batch_embeddings);
            return; // Success
        } catch (const std::exception &e) {
            if (retry == MAX_RETRIES - 1) {
                throw std::runtime_error("Failed after " + std::to_string(MAX_RETRIES) +
                                         " attempts: " + std::string(e.what()));
            }
            std::cerr << "Attempt " << retry + 1 << " failed: " << e.what() << std::endl;
        }
    }

    std::cerr << "Failed to process batch after " << MAX_RETRIES << " attempts" << std::endl;
    result_id = -1;
}

bool initializeSystem() {
    std::cout << "Initialize the system" << std::endl;

    // Initialize llama.cpp backend (required before model loading)
    // TODO: Consider where backend init/free should ideally live
    llama_backend_init(); 

    perform_similarity_check("/Users/manu/proj_tldr/tldr-dekstop/release-products/CosineSimilarityBatched.mlmodelc");

    if (!initializeDatabase()) {
        std::cerr << "Failed to initialize database" << std::endl;
        llama_backend_free(); // Clean up backend if db fails
        return false;
    }

    // Initialize LLM models via the manager
    if (!g_llm_manager.initialize_chat_model(CHAT_MODEL_PATH)) {
        std::cerr << "Failed to initialize chat model" << std::endl;
        // Database is initialized, but models failed. Cleanup backend.
        llama_backend_free();
        return false;
    }

    if (!g_llm_manager.initialize_embeddings_model(EMBEDDINGS_MODEL_PATH)) {
        std::cerr << "Failed to initialize embeddings model" << std::endl;
        // Chat model initialized, embeddings failed. Cleanup backend.
        // LlmManager destructor will handle chat model cleanup.
        llama_backend_free();
        return false;
    }

    // Initialize CURL globally (Restored)
    CURLcode curl_init_ret = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (curl_init_ret != CURLE_OK) {
        std::cerr << "Failed to initialize CURL: " << curl_easy_strerror(curl_init_ret) << std::endl;
        // Cleanup already initialized components
        llama_backend_free();
        // LlmManager destructor handles models/contexts
        // Database unique_ptr handles db connection
        return false;
    }

    std::cout << "System initialized successfully." << std::endl;
    return true;
}

void cleanupSystem() {
    // Cleanup CURL (If it was ever initialized - check initializeSystem)
    curl_global_cleanup();

    // Database is managed by unique_ptr, will clean up automatically.
    // g_db.reset();

    // LLM Manager cleans up models/contexts in its destructor when g_llm_manager goes out of scope.

    // Clean up llama.cpp backend
    llama_backend_free();
    std::cout << "System cleaned up." << std::endl;
}

void obtainEmbeddings(const std::vector<std::string> &chunks, size_t batch_size, size_t num_threads) {
    // System initialization is now handled by initializeSystem()
    const size_t total_batches = (chunks.size() + batch_size - 1) / batch_size;
    std::cout << "Processing " << chunks.size() << " chunks in " << total_batches
            << " batches using " << num_threads << " threads\n";
    std::vector<int> ids(num_threads, -1);

    try {
        for (size_t batch_start = 0; batch_start < chunks.size(); batch_start += batch_size * num_threads) {
            std::vector<std::thread> threads;

            // Launch threads
            for (size_t j = 0; j < num_threads && batch_start + j * batch_size < chunks.size(); ++j) {
                size_t start = batch_start + j * batch_size;
                size_t end = std::min(start + batch_size, chunks.size());

                threads.emplace_back(
                    [&chunks, start, end, batch_start, batch_size, total_batches, &ids, j, num_threads]() {
                        try {
                            // Process chunks directly without creating a copy
                            size_t batch_num = batch_start / (batch_size * num_threads) * num_threads + j;
                            processChunkBatch(std::vector<std::string_view>(
                                                  chunks.begin() + start,
                                                  chunks.begin() + end
                                              ), batch_num, total_batches, ids[j]);
                        } catch (const std::exception &e) {
                            std::cerr << "Thread " << j << " error: " << e.what() << std::endl;
                        }
                    });
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
    std::string doc_text = extractTextFromPDF(translatePath(sourcePath));
    if (doc_text.empty()) {
        std::cerr << "Error: No text extracted from PDF." << std::endl;
        return;
    }
    // print text length only
    std::cout << "Extracted text length: " << doc_text.length() << std::endl;
    const std::vector<std::string> chunks = splitTextIntoChunks(doc_text, MAX_CHUNK_SIZE, CHUNK_N_OVERLAP);
    std::cout << "Number of chunks: " << chunks.size() << std::endl;
    obtainEmbeddings(chunks,BATCH_SIZE,NUM_THREADS);
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

void queryRag(const std::string& user_query) {
    if (!g_db) {
        std::cerr << "Database not initialized" << std::endl;
        return;
    }

    try {
        // Get embeddings for the user query using LlmManager
        std::vector<std::string_view> query_vec = {user_query};
        std::vector<std::vector<float>> query_embeddings = g_llm_manager.get_embeddings(query_vec);

        if (query_embeddings.empty() || query_embeddings[0].empty()) {
            std::cerr << "Failed to get embeddings for the query." << std::endl;
            // Handle error - maybe return or throw
            return; 
        }

        // Find similar chunks in the database
        auto similar_chunks = g_db->searchSimilarVectors(query_embeddings[0], K_SIMILAR_CHUNKS_TO_RETRIEVE);

        // 3. Prepare context from similar chunks
        std::string context_str;
        for (const auto& [chunk, similarity] : similar_chunks) {
            context_str += chunk + "\n\n"; // Simple concatenation
        }

        if (context_str.empty()) {
            context_str = "No relevant context found.";
        }

        // 4. Generate response using LlmManager's chat model
        std::string final_response = g_llm_manager.get_chat_response(context_str, user_query);
        // Old way:

        // 5. Print results
        std::cout << "\nGenerated Response:\n";
        std::cout << final_response << "\n";
        std::cout << "\nContext used:\n";
        for (const auto& [chunk, similarity] : similar_chunks) {
            std::cout << "\nSimilarity: " << similarity << "\n";
            std::cout << "Content: " << chunk << "\n";
            std::cout << "----------------------------------------\n";
        }

    } catch (const std::exception &e) {
        std::cerr << "RAG Query error: " << e.what() << std::endl;
        // Handle error appropriately
    }
}

void command_loop() {
    std::string input;
    std::map<std::string, std::function<void(const std::string &)> > actions = {
        {"do-rag", doRag},
        {"add-corpus", addCorpus},
        {"delete-corpus", deleteCorpus},
        {"query", [](const std::string& query) { queryRag(query); }}
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
