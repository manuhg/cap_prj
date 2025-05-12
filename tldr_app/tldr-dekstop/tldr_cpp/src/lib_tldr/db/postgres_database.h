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
        int64_t saveEmbeddings(const std::vector<std::string_view> &chunks,
                               const json &embeddings_response,
                               const std::vector<uint64_t> &embedding_hashes = {}) override;
        bool getEmbeddings(int64_t id, std::vector<std::string> &chunks, json &embeddings) override;

        // Perform vector similarity search
        std::vector<std::tuple<std::string, float, uint64_t>> searchSimilarVectors(const std::vector<float>& query_vector, int k = 5) override;

        
        // Get text chunks by their hash values (for NPU-accelerated similarity search)
        std::map<uint64_t, std::string> getChunksByHashes(const std::vector<uint64_t>& hashes) override;

    private:
        std::string connection_string_;
        ConnectionPool<pqxx::connection> conn_pool;

        bool openConnection(pqxx::connection *&conn);
        void closeConnection(pqxx::connection *conn);
    };
}

#endif // TLDR_CPP_POSTGRES_DATABASE_H
