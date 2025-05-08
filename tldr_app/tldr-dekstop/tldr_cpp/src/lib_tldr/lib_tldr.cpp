#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pqxx/pqxx>
#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-page.h>
#include <thread>
#include <mutex>
#include <cstdlib>
#include <regex>
// #include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <functional> // Include for std::hash

#include "lib_tldr.h"
#include "llama.h"
#include "npu_accelerator.h"
#include "llm/llm-wrapper.h"
#include "lib_tldr/constants.h"

using json = nlohmann::json;

// Global database instance
std::unique_ptr<tldr::Database> g_db;

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

int64_t saveEmbeddingsToDb(const std::vector<std::string_view> &chunks, const std::vector<std::vector<float>> &embeddings, const std::vector<size_t> &embeddings_hash) {
    if (!g_db) {
        std::cerr << "Database not initialized" << std::endl;
        return -1;
    }

    // Convert embeddings to JSON format
    json embeddings_json;
    embeddings_json["embeddings"] = json::array();
    for(const auto& emb : embeddings) {
        embeddings_json["embeddings"].push_back(emb);
    }

    if (embeddings_json["embeddings"].empty()) {
        std::cerr << "No embeddings to save." << std::endl;
        return -1;
    }

    // Pass the computed embeddings_hash to the database
    return g_db->saveEmbeddings(chunks, embeddings_json, embeddings_hash);
}
/*
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
*/
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

// Helper: Compute hash for each embedding
static std::vector<size_t> computeEmbeddingHashes(const std::vector<std::vector<float>>& embeddings_list) {
    std::vector<size_t> hashes;
    hashes.reserve(embeddings_list.size());

    std::hash<float> float_hasher;
    for (const auto& emb : embeddings_list) {
        size_t seed = 0;
        for (float v : emb) {
            // Combine hash (similar to boost::hash_combine)
            seed ^= float_hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        hashes.push_back(seed);
    }
    return hashes;
}

// Save embeddings to database with thread safety
int saveEmbeddingsThreadSafe(const std::vector<std::string_view> &batch,
    const std::vector<std::vector<float>> &batch_embeddings, const std::vector<size_t> &embeddings_hash) {

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
    // Pass the raw embeddings and computed hashes to saveEmbeddingsToDb
    int64_t saved_id = saveEmbeddingsToDb(batch, batch_embeddings, embeddings_hash);
    if (saved_id < 0) {
        throw std::runtime_error("Failed to save embeddings to database");
    }
    return saved_id;
}
/*
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
            std::vector<std::vector<float>> batch_embeddings = tldr::get_llm_manager().get_embeddings(batch);
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
*/
bool initializeSystem() {
    std::cout << "Initialize the system" << std::endl;

    // Initialize llama.cpp backend (required before model loading)
    // TODO: Consider where backend init/free should ideally live

    // perform_similarity_check("/Users/manu/proj_tldr/tldr-dekstop/release-products/CosineSimilarityBatched.mlmodelc");
    if (!initializeDatabase()) {
        std::cerr << "Failed to initialize database" << std::endl;
        return false;
    }

    // Initialize CURL globally (Restored)
    // CURLcode curl_init_ret = curl_global_init(CURL_GLOBAL_DEFAULT);
    // if (curl_init_ret != CURLE_OK) {
        // std::cerr << "Failed to initialize CURL: " << curl_easy_strerror(curl_init_ret) << std::endl;
        // Cleanup already initialized components
        // LlmManager destructor will handle chat model cleanup.
        // Database unique_ptr handles db connection
        // return false;
    // }
    tldr::initialize_llm_manager_once();

    std::cout << "System initialized successfully." << std::endl;
    return true;
}

void cleanupSystem() {
    // Cleanup CURL (If it was ever initialized - check initializeSystem)
    // curl_global_cleanup();

    // Database is managed by unique_ptr, will clean up automatically.
    // g_db.reset();
    tldr::get_llm_manager().cleanup();

    std::cout << "System cleaned up." << std::endl;
}

// Vector dump functionality is now in vec_dump.h/cpp

std::pair<std::vector<std::vector<float>>, std::vector<size_t>>
obtainEmbeddings(const std::vector<std::string> &chunks, size_t batch_size, size_t num_threads) {
    // System initialization is now handled by initializeSystem()
    const size_t total_batches = (chunks.size() + batch_size - 1) / batch_size;
    std::cout << "Processing " << chunks.size() << " chunks in " << total_batches
            << " batches using " << num_threads << " threads\n";

    // We'll collect all embeddings and hashes
    std::vector<std::vector<float>> all_embeddings;
    std::vector<size_t> all_hashes;
    all_embeddings.reserve(chunks.size());
    all_hashes.reserve(chunks.size());

    std::vector<int> ids(num_threads, -1);

    try {
        for (size_t batch_start = 0; batch_start < chunks.size(); batch_start += batch_size * num_threads) {
            std::vector<std::thread> threads;
            std::vector<std::vector<std::vector<float>>> thread_embeddings(num_threads);
            std::vector<std::vector<size_t>> thread_hashes(num_threads);

            // Create a mutex to protect access to the embeddings and hashes collections
            // std::mutex collection_mutex;

            // Launch threads
            // for (size_t j = 0; j < num_threads && batch_start + j * batch_size < chunks.size(); ++j) {
                int j =0;
                size_t start = batch_start + j * batch_size;
                size_t end = std::min(start + batch_size, chunks.size());
/*
                threads.emplace_back(
                    [&chunks, start, end, batch_start, batch_size, total_batches, &ids, j, num_threads,
                     &thread_embeddings, &thread_hashes]() {*/
                        try {
                            // Create a vector of string_view for the current batch
                            std::vector<std::string_view> batch_chunks(
                                chunks.begin() + start,
                                chunks.begin() + end
                            );

                            // Process chunks directly without creating a copy
                            size_t batch_num = batch_start / (batch_size * num_threads) * num_threads + j;

                            // Get embeddings for this batch
                            std::vector<std::vector<float>> batch_emb = tldr::get_llm_manager().get_embeddings(batch_chunks);

                            // Compute hashes for these embeddings
                            std::vector<size_t> batch_hashes = computeEmbeddingHashes(batch_emb);

                            // Save the embeddings and hashes for this thread
                            thread_embeddings[j] = std::move(batch_emb);
                            thread_hashes[j] = std::move(batch_hashes);

                            saveEmbeddingsThreadSafe(batch_chunks,batch_emb,batch_hashes);

                            // Also save to database for consistency with previous behavior
                            // processChunkBatch(batch_chunks, batch_num, total_batches, ids[j]);

                        } catch (const std::exception &e) {
                            std::cerr << "Thread " << j << " error: " << e.what() << std::endl;
                        }
                    /* });
            }*/

            // Wait for all threads in this group to complete
            for (auto &thread: threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }

            // Collect all embeddings and hashes from this batch of threads
            for (size_t j = 0; j < num_threads; ++j) {
                if (!thread_embeddings[j].empty()) {
                    all_embeddings.insert(all_embeddings.end(),
                                         thread_embeddings[j].begin(),
                                         thread_embeddings[j].end());

                    all_hashes.insert(all_hashes.end(),
                                     thread_hashes[j].begin(),
                                     thread_hashes[j].end());
                }
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Error processing chunks: " << e.what() << std::endl;
        throw; // Re-throw to allow proper cleanup
    }

    return {all_embeddings, all_hashes};
}

void addCorpus(const std::string &sourcePath) {
    std::string expanded_path = translatePath(sourcePath);
    std::string doc_text = extractTextFromPDF(expanded_path);

    if (doc_text.empty()) {
        std::cerr << "Error: No text extracted from PDF." << std::endl;
        return;
    }

    // Print text length only
    std::cout << "Extracted text length: " << doc_text.length() << std::endl;
    const std::vector<std::string> chunks = splitTextIntoChunks(doc_text, MAX_CHUNK_SIZE, CHUNK_N_OVERLAP);
    std::cout << "Number of chunks: " << chunks.size() << std::endl;

    // Get embeddings and their hashes
    auto [embeddings, hashes] = obtainEmbeddings(chunks, BATCH_SIZE, NUM_THREADS);

    // Dump vectors and hashes to file for memory mapping
    tldr::dump_vectors_to_file(expanded_path, embeddings, hashes);

    std::cout << "Corpus added successfully." << std::endl;
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

void queryRag(const std::string& user_query, const std::string& corpus_dir) {
    if (!g_db) {
        std::cerr << "Database not initialized" << std::endl;
        return;
    }

    try {
        // Get embeddings for the user query using LlmManager
        std::vector<std::string_view> query_vec = {user_query};
        std::vector<std::vector<float>> query_embeddings = tldr::get_llm_manager().get_embeddings(query_vec);

        if (query_embeddings.empty() || query_embeddings[0].empty()) {
            std::cerr << "Failed to get embeddings for the query." << std::endl;
            // Handle error - maybe return or throw
            return;
        }

        std::cout << "Using NPU-accelerated similarity search..." << std::endl;

        // Use NPU-accelerated similarity search instead of database search
        auto similar_chunks = searchSimilarVectorsNPU(
            query_embeddings[0],         // Query vector
            corpus_dir,                  // Vector corpus directory
            K_SIMILAR_CHUNKS_TO_RETRIEVE // Number of results to return
        );

        if (similar_chunks.empty()) {
            std::cout << "No results from NPU search, falling back to database search..." << std::endl;
            // Fallback to traditional database search if NPU search returns no results
            similar_chunks = g_db->searchSimilarVectors(query_embeddings[0], K_SIMILAR_CHUNKS_TO_RETRIEVE);
        }

        // 3. Prepare context from similar chunks
        std::string context_str;
        for (const auto& [chunk, similarity] : similar_chunks) {
            context_str += chunk + "\n\n"; // Simple concatenation
        }

        if (context_str.empty()) {
            context_str = "No relevant context found.";
        }

        // 4. Generate response using LlmManager's chat model
        std::string final_response = tldr::get_llm_manager().get_chat_response(context_str, user_query);
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

// Wrapper function for NPU-accelerated vector similarity search
std::vector<std::pair<std::string, float>> searchSimilarVectorsNPU(
    const std::vector<float>& query_vector,
    const std::string& corpus_dir,
    int k
) {
    std::vector<std::pair<std::string, float>> results;
/*
    // Get CoreML model path from the release-products directory
    const char* model_path = "/Users/manu/proj_tldr/tldr-dekstop/release-products/artifacts/CosineSimilarityBatched.mlmodelc";

    // We'll collect the hashes from the results and only then query the database
    // This is more efficient than loading all embeddings upfront

    try {
        // Convert vector dimensions to int32_t
        int32_t dimensions = static_cast<int32_t>(query_vector.size());

        // Prepare to call Swift function
        int32_t result_count = 0;

        // Call the NPU similarity search function
        SimilarityResult* swift_results = retrieve_similar_vectors_from_corpus(
            model_path,                             // CoreML model path
            corpus_dir.c_str(),                     // Vector corpus directory
            query_vector.data(),                    // Query vector pointer
            dimensions,                             // Query vector dimensions
            k,                                      // Number of results to return
            &result_count                           // Output: number of results
        );

        if (swift_results && result_count > 0) {
            // Collect all the hashes we need to look up
            std::vector<uint64_t> hashes_to_lookup;
            std::map<uint64_t, float> hash_scores;

            for (int32_t i = 0; i < result_count; i++) {
                uint64_t hash = swift_results[i].hash;
                float score = swift_results[i].score;

                hashes_to_lookup.push_back(hash);
                hash_scores[hash] = score; // Store the score for this hash
            }

            // Query the database for just these specific hashes
            std::map<uint64_t, std::string> hash_to_text;
            if (g_db) {
                hash_to_text = g_db->getChunksByHashes(hashes_to_lookup);
            }

            // Convert results to the expected format using the looked-up text chunks
            for (const auto& [hash, score] : hash_scores) {
                auto it = hash_to_text.find(hash);
                if (it != hash_to_text.end()) {
                    results.emplace_back(it->second, score);
                } else {
                    // If text not found, use hash as identifier
                    std::string hash_id = "Hash_" + std::to_string(hash);
                    results.emplace_back(hash_id, score);
                }
            }

            // Free the results from Swift
            free(swift_results);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in NPU similarity search: " << e.what() << std::endl;
    }
*/
    return results;
}

// Test vector cache dump and read functionality
bool test_vector_cache() {
    std::cout << "=== Testing Vector Cache Dump and Read Functionality ===" << std::endl;

    // Create sample embeddings and hashes
    std::vector<std::vector<float>> test_embeddings;
    std::vector<size_t> test_hashes;

    // Create 5 test embeddings with 16 dimensions each
    const size_t num_embeddings = 5;
    const size_t dimensions = 16;

    std::cout << "Creating " << num_embeddings << " test embeddings with "
              << dimensions << " dimensions each" << std::endl;

    // Initialize with deterministic values for testing
    for (size_t i = 0; i < num_embeddings; i++) {
        std::vector<float> embedding(dimensions);
        for (size_t j = 0; j < dimensions; j++) {
            embedding[j] = static_cast<float>((i + 1) * 0.1f + j * 0.01f); // Deterministic pattern
        }
        test_embeddings.push_back(std::move(embedding));

        // Create corresponding hash
        test_hashes.push_back(1000000 + i * 10000); // Simple deterministic hash for testing
    }

    // Test file path
    std::string test_file = "vector_cache_test.bin";

    // Step 1: Dump the test data
    std::cout << "\nStep 1: Dumping test embeddings to " << test_file << std::endl;
    if (!tldr::dump_vectors_to_file(test_file, test_embeddings, test_hashes)) {
        std::cerr << "Error: Failed to dump test embeddings" << std::endl;
        return false;
    }

    // Step 2: Read the dumped file
    std::cout << "\nStep 2: Reading the vector dump file" << std::endl;
    auto mapped_data = tldr::read_vector_dump_file(test_file);
    if (!mapped_data) {
        std::cerr << "Error: Failed to read the vector dump file" << std::endl;
        return false;
    }

    // Step 3: Print info about the file
    print_vector_dump_info(mapped_data.get(), test_file, false);

    // Step 4: Verify contents
    std::cout << "\nStep 4: Verifying file contents" << std::endl;

    // Verify header
    bool header_verified =
        (mapped_data->header->num_entries == num_embeddings) &&
        (mapped_data->header->hash_size_bytes == sizeof(size_t)) &&
        (mapped_data->header->vector_dimensions == dimensions);

    std::cout << "Header verification: " << (header_verified ? "PASSED" : "FAILED") << std::endl;

    // Verify contents by checking values at index 1 (second element)
    bool data_verified = true;
    size_t test_idx = 1;

    if (test_idx < mapped_data->header->num_entries) {
        std::cout << "\nVerifying element at index " << test_idx << ":" << std::endl;

        // Original hash and the one read from file
        size_t original_hash = test_hashes[test_idx];
        size_t read_hash = mapped_data->hashes[test_idx];

        std::cout << "Hash verification: Original = " << original_hash
                  << ", Read = " << read_hash
                  << " -> " << (original_hash == read_hash ? "MATCH" : "MISMATCH") << std::endl;

        // Check a few dimensions of the embedding vector
        const float* read_vector = mapped_data->vectors + (test_idx * mapped_data->header->vector_dimensions);

        std::cout << "Vector verification (first 5 dimensions):" << std::endl;
        bool vector_matches = true;
        for (size_t i = 0; i < 5 && i < mapped_data->header->vector_dimensions; i++) {
            float original_val = test_embeddings[test_idx][i];
            float read_val = read_vector[i];
            bool matches = (std::abs(original_val - read_val) < 0.000001f); // Floating point comparison with epsilon

            std::cout << "  Dim " << i << ": Original = " << original_val
                      << ", Read = " << read_val
                      << " -> " << (matches ? "MATCH" : "MISMATCH") << std::endl;

            if (!matches) vector_matches = false;
        }

        data_verified = (original_hash == read_hash) && vector_matches;
    }

    // Print final result
    std::cout << "\nTest result: " << (header_verified && data_verified ? "PASSED" : "FAILED") << std::endl;

    return header_verified && data_verified;
}

void command_loop() {
    std::string input;
    std::map<std::string, std::function<void(const std::string &)> > actions = {
        {"do-rag", doRag},
        {"add-corpus", addCorpus},
        {"delete-corpus", deleteCorpus},
        {"query", [](const std::string& query) { queryRag(query); }},
        {"read-vectors", [](const std::string& path) {
            auto data = tldr::read_vector_dump_file(path);
            if (data) {
                tldr::print_vector_dump_info(data.get(), path, true);
            } else {
                std::cerr << "Failed to read vector file: " << path << std::endl;
            }
        }},
        {"test-vectors", [](const std::string&) { tldr::test_vector_cache(); }}
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
