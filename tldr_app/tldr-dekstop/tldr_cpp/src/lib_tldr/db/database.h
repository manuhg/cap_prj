#ifndef TLDR_CPP_DATABASE_H
#define TLDR_CPP_DATABASE_H

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "../constants.h"
using json = nlohmann::json;

namespace tldr {
    class Database {
    public:
        virtual ~Database() = default;

        // Initialize the database, create tables if needed
        virtual bool initialize() = 0;

        // Save embeddings to the database
        virtual int64_t saveEmbeddings(const std::vector<std::string_view> &chunks, 
                                      const json &embeddings_response,
                                      const std::vector<size_t> &embedding_hashes = {}) = 0;

        // Get embeddings by ID
        virtual bool getEmbeddings(int64_t id, std::vector<std::string> &chunks, json &embeddings) = 0;

        virtual std::vector<std::pair<std::string, float>> searchSimilarVectors(const std::vector<float>& query_vector, int k) = 0;
        
        // Get all embeddings from the database (for NPU-accelerated similarity search)
        virtual std::map<std::string, std::vector<float>> getAllEmbeddings() = 0;
        
        // Get text chunks by hash values
        virtual std::map<uint64_t, std::string> getChunksByHashes(const std::vector<uint64_t>& hashes) = 0;
    };
}

#endif // TLDR_CPP_DATABASE_H
