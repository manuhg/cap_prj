//
//

#ifndef TLDR_CPP_MAIN_H
#define TLDR_CPP_MAIN_H

#include <nlohmann/json.hpp>
#include "db/database.h"
#include "db/sqlite_database.h"
#include "db/postgres_database.h"

#define PAGE_DELIMITER "\n\n"
#define CHUNK_N_SENTENCES 10
#define CHUNK_N_OVERLAP 20 // overlap in characters at start and end

#define LLM_EMBEDDINGS_URL "http://localhost:8080/embeddings"
#define DB_PATH "embeddings.db"
#define PG_CONNECTION "dbname=tldr_embeddings user=postgres password=postgres host=localhost port=5432"
// tests
void test_extractTextFromPDF();
//
std::string extractTextFromPDF(const std::string &filename);
void doRag(const std::string &conversationId);
void addCorpus(const std::string &sourcePath);
void deleteCorpus(const std::string &corpusId);
void command_loop();

struct embeddings_request {
    std::vector<std::string> input;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(embeddings_request, input)
#endif //TLDR_CPP_MAIN_H
