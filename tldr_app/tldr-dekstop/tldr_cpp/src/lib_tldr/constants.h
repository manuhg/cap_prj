#ifndef TLDR_CPP_CONSTANTS_H
#define TLDR_CPP_CONSTANTS_H
#include <nlohmann/json.hpp>
#include <vector>

// Text processing constants
#define PAGE_DELIMITER "\n\n"
#define AVG_WORDS_PER_SENTENCE 6
#define AVG_CHARS_PER_WORD 5
#define CHUNK_N_SENTENCES 10
#define CHUNK_N_OVERLAP 80 // overlap in characters at start and end
#define CHUNK_N_CHARS ((CHUNK_N_SENTENCES * AVG_WORDS_PER_SENTENCE * AVG_CHARS_PER_WORD)+(CHUNK_N_OVERLAP*2))
#define MAX_CHARS_PER_BATCH 512
#define MAX_CHUNK_SIZE (MAX_CHARS_PER_BATCH-(CHUNK_N_OVERLAP*2))
#define BATCH_SIZE 8

#define NUM_THREADS 4
#define CONN_POOL_SIZE NUM_THREADS

// Database constants
#define USE_POSTGRES true  // Set to true to use PostgreSQL, false for SQLite
#define DB_PATH "~/proj_tldr/datastore/embeddings.db"
#define PG_CONNECTION "dbname=tldr user=postgres password=postgres host=localhost port=5432"

// HTTP request constants
#define EMBEDDINGS_URL "http://localhost:8084/embeddings"
#define CHAT_URL "http://localhost:8088/v1/chat/completions"

#define EMBEDDINGS_MODEL_PATH "/Users/manu/llm-weights/Llama-3.2-1B-Instruct-Q3_K_L-lms.gguf"
#define CHAT_MODEL_PATH "/Users/manu/llm-weights/embedding/all-MiniLM-L6-v2-ggml-model-f16.gguf"

#define CONNECT_TIMEOUT_SECONDS 5
#define REQUEST_TIMEOUT_SECONDS 30
#define MAX_RETRIES 1
#define RETRY_DELAY_MS 1000
#define EMBEDDING_SIZE "384" // keep it string so that it can be inserted into create table stmt
#define K_SIMILAR_CHUNKS_TO_RETRIEVAL 5
struct embeddings_request {
    std::vector<std::string> input;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(embeddings_request, input)
#endif //TLDR_CPP_CONSTANTS_H
