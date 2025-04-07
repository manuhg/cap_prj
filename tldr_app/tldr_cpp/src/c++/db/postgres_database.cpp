#include "postgres_database.h"
#include <iostream>

namespace tldr {
    PostgresDatabase::PostgresDatabase(const std::string& connection_string)
        : connection_string_(connection_string) {}

    PostgresDatabase::~PostgresDatabase() {
        closeConnection();
    }

    bool PostgresDatabase::openConnection() {
        if (conn_) {
            return true;
        }

        try {
            conn_ = std::make_unique<pqxx::connection>(connection_string_);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Connection error: " << e.what() << std::endl;
            return false;
        }
    }

    void PostgresDatabase::closeConnection() {
        conn_.reset();
    }

    bool PostgresDatabase::initialize() {
        if (!openConnection()) {
            return false;
        }
        //Note: expects that the db is already setup, configured and running

        try {
            pqxx::work txn(*conn_);
            
            txn.exec(R"(
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            )");
            
            txn.commit();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Database initialization error: " << e.what() << std::endl;
            return false;
        }
    }

    int64_t PostgresDatabase::saveEmbeddings(const std::vector<std::string>& chunks, const json& embeddings_response) {
        if (!openConnection()) {
            return -1;
        }

        try {
            pqxx::work txn(*conn_);
            int64_t last_id = -1;

            // Insert each chunk with its embedding
            for (const auto& chunk : chunks) {
                pqxx::result r = txn.exec_params(
                    "INSERT INTO embeddings (chunk_text, embedding_data) VALUES ($1, $2) RETURNING id",
                    chunk,
                    embeddings_response.dump()
                );
                
                if (!r.empty()) {
                    last_id = r[0][0].as<int64_t>();
                }
            }

            txn.commit();
            return last_id;
        } catch (const std::exception& e) {
            std::cerr << "Error saving embeddings: " << e.what() << std::endl;
            return -1;
        }
    }

    bool PostgresDatabase::getEmbeddings(int64_t id, std::vector<std::string>& chunks, json& embeddings) {
        if (!openConnection()) {
            return false;
        }

        try {
            pqxx::work txn(*conn_);
            
            pqxx::result r = txn.exec_params(
                "SELECT chunk_text, embedding_data FROM embeddings WHERE id = $1",
                id
            );

            if (!r.empty()) {
                chunks.push_back(r[0]["chunk_text"].as<std::string>());
                embeddings = json::parse(r[0]["embedding_data"].as<std::string>());
                
                txn.commit();
                return true;
            }

            txn.commit();
            return false;
        } catch (const std::exception& e) {
            std::cerr << "Error retrieving embeddings: " << e.what() << std::endl;
            return false;
        }
    }
}
