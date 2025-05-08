#ifndef TLDR_CPP_SQLITE_DATABASE_H
#define TLDR_CPP_SQLITE_DATABASE_H

#include "database.h"
#include "connection_pool.h"
#include <sqlite3.h>

namespace tldr {
    class SQLiteDatabase : public Database {
    public:
        explicit SQLiteDatabase(const std::string &db_path);
        ~SQLiteDatabase() override;

        bool initialize() override;
        int64_t saveEmbeddings(const std::vector<std::string_view> &chunks, const json &embeddings_response, const std::vector<size_t> &embedding_hashes = {}) override;
        bool getEmbeddings(int64_t id, std::vector<std::string> &chunks, json &embeddings) override;
        std::vector<std::pair<std::string, float>> searchSimilarVectors(const std::vector<float>& query_vector, int k = 5) override;
        std::map<uint64_t, std::string> getChunksByHashes(const std::vector<uint64_t>& hashes) override;

    private:
        std::string db_path_;
        ConnectionPool<sqlite3> conn_pool;

        bool openConnection(sqlite3 *&db);
        void closeConnection(sqlite3 *db);
    };
}

#endif // TLDR_CPP_SQLITE_DATABASE_H
