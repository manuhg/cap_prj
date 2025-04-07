#ifndef TLDR_CPP_POSTGRES_DATABASE_H
#define TLDR_CPP_POSTGRES_DATABASE_H

#include "database.h"
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
        std::unique_ptr<pqxx::connection> conn_;
        
        bool openConnection();
        void closeConnection();
    };
}

#endif // TLDR_CPP_POSTGRES_DATABASE_H
