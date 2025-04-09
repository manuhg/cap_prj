#ifndef TLDR_CPP_POSTGRES_DATABASE_H
#define TLDR_CPP_POSTGRES_DATABASE_H

#include "database.h"
#include "connection_pool.h"
#include <pqxx/pqxx>

namespace tldr {
    class PostgresDatabase : public Database {
    public:
        explicit PostgresDatabase(const std::string& connection_string);
        ~PostgresDatabase() override;
        
        bool initialize() override;
        int64_t saveEmbeddings(const std::vector<std::string>& chunks, const json& embeddings_response) override;
        bool getEmbeddings(int64_t id, std::vector<std::string>& chunks, json& embeddings) override;
        
    private:
        std::string connection_string_;
        ConnectionPool<pqxx::connection> conn_pool;
        
        bool openConnection(pqxx::connection*& conn);
        void closeConnection(pqxx::connection* conn);
    };
}

#endif // TLDR_CPP_POSTGRES_DATABASE_H
