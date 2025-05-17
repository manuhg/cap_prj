#ifndef TLDR_CPP_SQLITE_DATABASE_H
#define TLDR_CPP_SQLITE_DATABASE_H

#include "database.h"
#include <SQLiteCpp/SQLiteCpp.h>
#include <string>
#include <vector>
#include <map>
#include <tuple>

namespace tldr {
    class SQLiteDatabase : public Database {
    public:
        explicit SQLiteDatabase(const std::string &db_path);
        ~SQLiteDatabase() override;

        bool initialize() override;
        
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
        int64_t saveEmbeddings(
            const std::vector<std::string_view> &chunks,
            const json &embeddings_response,
            const std::vector<uint64_t> &embedding_hashes,
            const std::vector<int>& chunkPageNums,
            const std::string& fileHash) override;
            
        bool getEmbeddings(int64_t id, std::vector<std::string> &chunks, json &embeddings) override;
        std::vector<std::tuple<std::string, float, uint64_t>> searchSimilarVectors(
            const std::vector<float>& query_vector, int k = 5) override;
            
        std::map<uint64_t, std::string> getChunksByHashes(const std::vector<uint64_t>& hashes) override;
        bool deleteEmbeddings(const std::string& file_hash) override;

    private:
        std::string db_path_;

        // These methods are kept for backward compatibility but are no longer used
        bool openConnection(sqlite3*& db);
        void closeConnection(sqlite3* db);
    };
}

#endif // TLDR_CPP_SQLITE_DATABASE_H
