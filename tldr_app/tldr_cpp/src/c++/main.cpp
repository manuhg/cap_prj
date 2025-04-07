#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <pqxx/pqxx>
#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-page.h>

#include <main.h>
#include <regex>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <memory>

using json = nlohmann::json;

void test_extractTextFromPDF() {
    std::string testFile = "../../corpus/current/97-things-every-software-architect-should-know.pdf";

    std::string extractedText = extractTextFromPDF(testFile);
    std::cout << extractedText.length() << std::endl;

    assert(extractedText.length()==268979);
    std::cout << "test_extractTextFromPDF passed!" << std::endl;
    std::cout << "Extracted text (first 50 chars): \n=========" << extractedText.substr(0, 50) << "\n=========" <<std::endl;
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
            for (unsigned char c : utf8_data) {
                if (c < 128) { // ASCII range
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
std::vector<std::string> splitTextIntoChunks(const std::string &text, size_t max_chunk_size = 2000, size_t overlap = 20) {
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


// Global database instance
std::unique_ptr<tldr::Database> g_db;

bool initializeDatabase(bool use_postgres = false) {
    if (!g_db) {
        if (use_postgres) {
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

void processChunkBatch(const std::vector<std::string>& batch, size_t batch_num, size_t total_batches) {
    // Create JSON array of texts
    json request = {
        {"input", batch}
    };
    
    std::cout << "Processing batch " << batch_num + 1 
              << " of " << total_batches << std::endl;

    // Convert JSON to string
    std::string json_str = request.dump();
    
    // Create a cURL handle
    CURL *curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }
    
    // Set up headers
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    // Response data will be stored here
    std::string response_data;
    
    // Set up cURL options
    curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8080/embeddings");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, +[](char *ptr, size_t size, size_t nmemb, void *userdata) {
                std::string *response = static_cast<std::string*>(userdata);
                response->append(ptr, size * nmemb);
                return size * nmemb;
            });
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
            
            res = curl_easy_perform(curl);
            
            if (res != CURLE_OK) {
                std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
                return std::vector<std::vector<float>>();
            }
            
            try {
                json response_json = json::parse(response);
                
                if (response_json.contains("error")) {
                    std::cerr << "API Error: " << response_json["error"]["message"] << std::endl;
                    return std::vector<std::vector<float>>();
                }
                
                if (!response_json.contains("data") || !response_json["data"].is_array()) {
                    std::cerr << "Invalid response format: missing 'data' array" << std::endl;
                    return std::vector<std::vector<float>>();
                }
                
                for (const auto& item : response_json["data"]) {
                    if (item.contains("embedding") && item["embedding"].is_array()) {
                        std::vector<float> embedding = item["embedding"].get<std::vector<float>>();
                        all_embeddings.push_back(embedding);
                    }
                }
            } catch (const json::parse_error& e) {
                std::cerr << "JSON parse error: " << e.what() << std::endl;
                return std::vector<std::vector<float>>();
            }
            
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
        }
        curl_global_cleanup();
    }
    
    return all_embeddings;
}

void sendEmbeddingsRequest(const std::vector<std::string> &chunks, bool use_postgres = false) {
    CURL *curl;
    CURLcode res;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        std::string url = LLM_EMBEDDINGS_URL;
        // convert chunks into json data
        embeddings_request request =  {chunks};
        json request_json = request;


        std::string jsonData = request_json.dump(2);
        std::cout <<"\njsonData:\n"<< jsonData.substr(0,100) << std::endl;

        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());

        // Set up response handling
        std::string response;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, +[](char *ptr, size_t size, size_t nmemb, void *userdata) {
            std::string *response = static_cast<std::string*>(userdata);
            response->append(ptr, size * nmemb);
            return size * nmemb;
        });
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            // size_t batch_size = MAX_CHARS_PER_BATCH/CHUNK_N_CHARS;
            std::vector<std::vector<float>> all_embeddings = processChunksInBatches(chunks);
            
            if (all_embeddings.empty()) {
                std::cerr << "Failed to process chunks" << std::endl;
                return;
            }
            
            // Create a new JSON object with just the embeddings array
            json embeddings_array;
            embeddings_array["embeddings"] = all_embeddings;
            
            size_t num_embeddings = embeddings_array["embeddings"].size();
            std::cout << "Processed " << num_embeddings << " embeddings for " << chunks.size() << " chunks" << std::endl;
            
            // Verify number of embeddings matches number of chunks
            if (num_embeddings != chunks.size()) {
                std::cerr << "Error: Number of embeddings (" << num_embeddings << ") does not match number of chunks (" << chunks.size() << ")" << std::endl;
                return;
            }
            
            // Initialize and save to database
            if (initializeDatabase(use_postgres)) {
                int64_t saved_id = saveEmbeddings(chunks, embeddings_array);
                if (saved_id != -1) {
                    std::cout << "Saved embeddings with ID: " << saved_id << std::endl;
                } else {
                    std::cerr << "Failed to save embeddings to database" << std::endl;
                }
            } else {
                std::cerr << "Failed to initialize database" << std::endl;
            }
        }
        // Clean up
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    curl_global_cleanup();
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
    sendEmbeddingsRequest(chunks);
    //TODO save the json to database and return the id
}

void deleteCorpus(const std::string &corpusId) {
    // Implement the function
    std::cout << "DELETE_CORPUS action with corpus_id: " << corpusId << std::endl;
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
    std::string testFile = "../corpus/current/97-things-every-software-architect-should-know.pdf";
    
    std::cout << "Testing addCorpus with file: " << testFile << std::endl;
    addCorpus(testFile);
    
    return 0;
}
