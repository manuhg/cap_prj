#include "postgres_database.h"
#include <iostream>
#include <atomic>
#include <pqxx/pqxx>

namespace tldr {
    PostgresDatabase::PostgresDatabase(const std::string& connection_string) 
        : connection_string_(connection_string), 
          conn_pool(
              connection_string, 
              CONN_POOL_SIZE,
              [](const std::string& conn_str) { 
                  return new pqxx::connection(conn_str); 
              }, 
              [](pqxx::connection* conn) { 
                  if (conn) {
                      try {
                          conn->close();
                      } catch (const std::exception& e) {
                          std::cerr << "Error closing connection: " << e.what() << std::endl;
                      }
                      delete conn;
                  }
              }
          ) {}

    PostgresDatabase::~PostgresDatabase() {}

    bool PostgresDatabase::openConnection(pqxx::connection*& conn) {
        try {
            conn = conn_pool.acquire();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to acquire connection: " << e.what() << std::endl;
            return false;
        }
    }

    void PostgresDatabase::closeConnection(pqxx::connection* conn) {
        if (conn) {
            conn_pool.release(conn);
        }
    }

    bool PostgresDatabase::initialize() {
        pqxx::connection* conn = nullptr;
        if (!openConnection(conn)) {
            return false;
        }

        try {
            pqxx::work txn(*conn);

            // Create embeddings table if not exists
            txn.exec(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "id BIGSERIAL PRIMARY KEY,"
                "chunk_text TEXT NOT NULL,"
                "embedding_data JSONB NOT NULL,"
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            );

            txn.commit();
            closeConnection(conn);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Initialization error: " << e.what() << std::endl;
            closeConnection(conn);
            return false;
        }
    }

    int64_t PostgresDatabase::saveEmbeddings(const std::vector<std::string>& chunks, const json& embeddings_response) {
        pqxx::connection* conn = nullptr;
        if (!openConnection(conn)) {
            return -1;
        }

        try {
            pqxx::work txn(*conn);

            // Prepare the insert statement with a unique name to avoid conflicts in multithreaded environment
            static std::atomic<int> statement_counter{0};
            std::string stmt_name = "insert_embedding_" + std::to_string(statement_counter++);
            
            // Prepare the statement with a unique name
            txn.conn().prepare(
                stmt_name, 
                "INSERT INTO embeddings (chunk_text, embedding_data) VALUES ($1, $2) RETURNING id"
            );

            int64_t last_id = -1;
            for (const auto& chunk : chunks) {
                // Execute the prepared statement with parameters
                pqxx::params params;
                params.append(chunk);
                params.append(embeddings_response.dump());
                auto result = txn.exec_prepared(stmt_name, params);

                // Get the last inserted id
                if (!result.empty()) {
                    last_id = result[0][0].as<int64_t>();
                }
            }

            txn.commit();
            closeConnection(conn);
            return last_id;
        } catch (const std::exception& e) {
            std::cerr << "Insertion error: " << e.what() << std::endl;
            closeConnection(conn);
            return -1;
        }
    }

    bool PostgresDatabase::getEmbeddings(int64_t id, std::vector<std::string>& chunks, json& embeddings) {
        pqxx::connection* conn = nullptr;
        if (!openConnection(conn)) {
            return false;
        }

        try {
            pqxx::work txn(*conn);

            // Create params for the query
            pqxx::params params;
            params.append(id);
            
            // Execute the query with parameters
            auto result = txn.exec_params(
                "SELECT chunk_text, embedding_data FROM embeddings WHERE id = $1",
                params
            );

            if (!result.empty()) {
                chunks.push_back(result[0]["chunk_text"].as<std::string>());
                embeddings = json::parse(result[0]["embedding_data"].as<std::string>());
                
                closeConnection(conn);
                return true;
            }

            closeConnection(conn);
            return false;
        } catch (const std::exception& e) {
            std::cerr << "Retrieval error: " << e.what() << std::endl;
            closeConnection(conn);
            return false;
        }
    }
}
