#ifndef TLDR_CPP_POSTGRES_DATABASE_H
#define TLDR_CPP_POSTGRES_DATABASE_H

#include "database.h"
#include "connection_pool.h"
#include <pqxx/pqxx>

namespace tldr {
    class PostgresDatabase : public Database {
    public:
        explicit PostgresDatabase(const std::string &connection_string);
        ~PostgresDatabase() override;

        bool initialize() override;
        int64_t saveEmbeddings(
            const std::vector<std::string_view> &chunks,
            const json &embeddings_response,
            const std::vector<uint64_t> &embedding_hashes,
            const std::vector<int> &chunk_page_nums,
            const std::string &file_hash) override;
        
        // New method to save embeddings with an existing connection
        int64_t saveEmbeddingsWithConnection(
            pqxx::connection* conn,
            const std::vector<std::string_view> &chunks,
            const json &embeddings_response,
            const std::vector<uint64_t> &embedding_hashes,
            const std::vector<int> &chunk_page_nums,
            const std::string &file_hash);

        bool getEmbeddings(int64_t id, std::vector<std::string> &chunks, json &embeddings) override;

        // Perform vector similarity search
        std::vector<std::tuple<std::string, float, uint64_t>> searchSimilarVectors(const std::vector<float>& query_vector, int k = 5) override;

        // Get text chunks by their hash values (for NPU-accelerated similarity search)
        std::map<uint64_t, std::string> getChunksByHashes(const std::vector<uint64_t>& hashes) override;

        // Save or update document metadata
        bool saveDocumentMetadata(
            const std::string& fileHash,
            const std::string& filePath,
            const std::string& fileName,
            const std::string& title,
            const std::string& author,
            const std::string& subject,
            const std::string& keywords,
            const std::string& creator,
            const std::string& producer,
            int pageCount) override;
            
        // Delete all embeddings for a specific file hash
        bool deleteEmbeddings(const std::string& file_hash) override;

        // Method to directly acquire a connection from the pool
        pqxx::connection* acquireConnection();
        
        // Method to release a connection back to the pool
        void releaseConnection(pqxx::connection* conn);

    private:
        std::string connection_string_;
        ConnectionPool<pqxx::connection> conn_pool;

        bool openConnection(pqxx::connection *&conn);
        void closeConnection(pqxx::connection *conn);
    };
}

#endif // TLDR_CPP_POSTGRES_DATABASE_H
