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
        virtual int64_t saveEmbeddings(
            const std::vector<std::string_view> &chunks,
            const json &embeddings_response,
            const std::vector<uint64_t> &embedding_hashes,
            const std::vector<int> &chunk_page_nums,
            const std::string &file_hash) = 0;

        // Get embeddings by ID
        virtual bool getEmbeddings(int64_t id, std::vector<std::string> &chunks, json &embeddings) = 0;

        virtual std::vector<CtxChunkMeta> searchSimilarVectors(const std::vector<float>& query_vector, int k) = 0;

        // Get text chunks by hash values with document metadata
        virtual std::map<uint64_t, CtxChunkMeta> getChunksByHashes(const std::vector<uint64_t>& hashes) = 0;

        // Save or update document metadata
        virtual bool saveDocumentMetadata(
            const std::string& fileHash,
            const std::string& filePath,
            const std::string& fileName,
            const std::string& title,
            const std::string& author,
            const std::string& subject,
            const std::string& keywords,
            const std::string& creator,
            const std::string& producer,
            int pageCount) = 0;
            
        // Delete all embeddings for a specific file hash
        virtual bool deleteEmbeddings(const std::string& file_hash) = 0;
    };
}

#endif // TLDR_CPP_DATABASE_H
