#include "postgres_database.h"
#include <iostream>
#include <atomic>
#include <pqxx/pqxx>
#include "../constants.h"
namespace tldr {
    PostgresDatabase::PostgresDatabase(const std::string &connection_string)
        : connection_string_(connection_string),
          conn_pool(
              connection_string,
              CONN_POOL_SIZE,
              [](const std::string &conn_str) {
                  return new pqxx::connection(conn_str);
              },
              [](pqxx::connection *conn) {
                  if (conn) {
                      try {
                          conn->close();
                      } catch (const std::exception &e) {
                          std::cerr << "Error closing connection: " << e.what() << std::endl;
                      }
                      delete conn;
                  }
              }
          ) {
    }

    PostgresDatabase::~PostgresDatabase() {
    }

    bool PostgresDatabase::openConnection(pqxx::connection *&conn) {
        try {
            conn = conn_pool.acquire();
            return true;
        } catch (const std::exception &e) {
            std::cerr << "Failed to acquire connection: " << e.what() << std::endl;
            return false;
        }
    }

    void PostgresDatabase::closeConnection(pqxx::connection *conn) {
        if (conn) {
            conn_pool.release(conn);
        }
    }

    bool PostgresDatabase::initialize() {
        pqxx::connection *conn = nullptr;
        if (!openConnection(conn)) {
            return false;
        }

        try {
            pqxx::work txn(*conn);

            // Enable required extensions
            txn.exec("CREATE EXTENSION IF NOT EXISTS vector");

            // Create embeddings table with vector column
            txn.exec(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "id BIGSERIAL PRIMARY KEY,"
                "chunk_text TEXT NOT NULL,"
                "text_hash BIGINT NOT NULL,"
                "embedding_hash BIGINT,"
                "embedding vector(" EMBEDDING_SIZE ") NOT NULL,"  // Assuming 2048-dimensional embeddings
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            );

            // Create unique index on text_hash to prevent duplicates
            txn.exec("CREATE UNIQUE INDEX IF NOT EXISTS embeddings_text_hash_idx ON embeddings (text_hash)");

            // Create index for vector similarity search
            txn.exec("CREATE INDEX IF NOT EXISTS embeddings_vector_idx ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)");

            txn.commit();
            closeConnection(conn);
            return true;
        } catch (const std::exception &e) {
            std::cerr << "Initialization error: " << e.what() << std::endl;
            closeConnection(conn);
            return false;
        }
    }

    int64_t PostgresDatabase::saveEmbeddings(const std::vector<std::string> &chunks, const json &embeddings_response, const std::vector<size_t> &embedding_hashes) {
        pqxx::connection *conn = nullptr;
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
                "INSERT INTO embeddings (chunk_text, text_hash, embedding_hash, embedding) "
                "VALUES ($1, hashtextextended($1::text, 0), $2, $3) "
                "ON CONFLICT (text_hash) DO NOTHING "
                "RETURNING id"
            );

            int64_t last_id = -1;
            // Get the embeddings array from the response
            const auto& embeddings = embeddings_response["embeddings"];
            
            // Insert each chunk and its embedding
            for (size_t i = 0; i < chunks.size(); ++i) {
                // Convert embedding vector to PostgreSQL array string
                std::string vector_str = "[";
                for (size_t j = 0; j < embeddings[i].size(); ++j) {
                    if (j > 0) vector_str += ",";
                    vector_str += std::to_string(embeddings[i][j].get<float>());
                }
                vector_str += "]";

                // Get hash if available, or use 0 as default
                size_t hash = i < embedding_hashes.size() ? embedding_hashes[i] : 0;
                
                // Execute the prepared statement with parameters
                pqxx::params params;
                params.append(chunks[i]);
                params.append(static_cast<long long>(hash));  // Cast to long long for PostgreSQL bigint
                params.append(vector_str);
                auto result = txn.exec_prepared(stmt_name, params);

                // Get the last inserted id
                if (!result.empty()) {
                    last_id = result[0][0].as<int64_t>();
                }
            }

            txn.commit();
            closeConnection(conn);
            return last_id;
        } catch (const std::exception &e) {
            std::cerr << "Insertion error: " << e.what() << std::endl;
            closeConnection(conn);
            return -1;
        }
    }

    bool PostgresDatabase::getEmbeddings(int64_t id, std::vector<std::string> &chunks, json &embeddings) {
        pqxx::connection *conn = nullptr;
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
        } catch (const std::exception &e) {
            std::cerr << "Retrieval error: " << e.what() << std::endl;
            closeConnection(conn);
            return false;
        }
    }

    std::vector<std::pair<std::string, float>> PostgresDatabase::searchSimilarVectors(const std::vector<float>& query_vector, int k) {
        pqxx::connection *conn = nullptr;
        if (!openConnection(conn)) {
            return {};
        }

        try {
            pqxx::work txn(*conn);
            
            // Convert vector to PostgreSQL array string
            std::string vector_str = "ARRAY[";
            for (size_t i = 0; i < query_vector.size(); ++i) {
                if (i > 0) vector_str += ",";
                vector_str += std::to_string(query_vector[i]);
            }
            vector_str += "]::vector";

            // Perform similarity search using cosine distance
            std::string query = 
                "SELECT chunk_text, 1 - (embedding <=> " + vector_str + ") as similarity "
                "FROM embeddings "
                "ORDER BY embedding <=> " + vector_str + " "
                "LIMIT " + std::to_string(k);

            auto result = txn.exec(query);
            std::vector<std::pair<std::string, float>> results;

            for (const auto& row : result) {
                results.emplace_back(
                    row["chunk_text"].as<std::string>(),
                    row["similarity"].as<float>()
                );
            }

            txn.commit();
            closeConnection(conn);
            return results;
        } catch (const std::exception &e) {
            std::cerr << "Search error: " << e.what() << std::endl;
            closeConnection(conn);
            return {};
        }
    }
}
