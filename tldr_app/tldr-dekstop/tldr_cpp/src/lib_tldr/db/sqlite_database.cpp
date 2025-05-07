#include "sqlite_database.h"
#include <SQLiteCpp/SQLiteCpp.h>
#include <iostream>
#include <sstream>

namespace tldr {
    SQLiteDatabase::SQLiteDatabase(const std::string &db_path)
        : db_path_(db_path),
          conn_pool(
              db_path,
              CONN_POOL_SIZE,
              [](const std::string &path) {
                  sqlite3 *db = nullptr;
                  int rc = sqlite3_open(path.c_str(), &db);
                  if (rc) {
                      throw std::runtime_error("Can't open database: " + std::string(sqlite3_errmsg(db)));
                  }
                  return db;
              },
              [](sqlite3 *db) {
                  if (db) {
                      sqlite3_close(db);
                  }
              }
          ) {
    }

    SQLiteDatabase::~SQLiteDatabase() {
    }

    bool SQLiteDatabase::openConnection(sqlite3 *&db) {
        try {
            db = conn_pool.acquire();
            return true;
        } catch (const std::exception &e) {
            std::cerr << "Failed to acquire connection: " << e.what() << std::endl;
            return false;
        }
    }

    void SQLiteDatabase::closeConnection(sqlite3 *db) {
        if (db) {
            conn_pool.release(db);
        }
    }

    bool SQLiteDatabase::initialize() {
        sqlite3 *db = nullptr;
        if (!openConnection(db)) {
            return false;
        }

        char *errMsg = nullptr;
        int rc;

        // Enable WAL mode for better performance and concurrency
        rc = sqlite3_exec(db, "PRAGMA journal_mode=WAL", 0, 0, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "Failed to enable WAL mode: " << errMsg << std::endl;
            sqlite3_free(errMsg);
            closeConnection(db);
            return false;
        }

        // Set other recommended WAL-related PRAGMAs
        rc = sqlite3_exec(db, "PRAGMA synchronous=NORMAL", 0, 0, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "Failed to set synchronous mode: " << errMsg << std::endl;
            sqlite3_free(errMsg);
            closeConnection(db);
            return false;
        }

        const char *sql = "CREATE TABLE IF NOT EXISTS embeddings ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "chunk_text TEXT NOT NULL,"
                "embedding_hash INTEGER,"
                "embedding_data TEXT NOT NULL,"
                "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)";

        rc = sqlite3_exec(db, sql, 0, 0, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "SQL error: " << errMsg << std::endl;
            sqlite3_free(errMsg);
            closeConnection(db);
            return false;
        }

        closeConnection(db);
        return true;
    }

    int64_t SQLiteDatabase::saveEmbeddings(const std::vector<std::string_view> &chunks, const json &embeddings_response, const std::vector<size_t> &embedding_hashes) {
        sqlite3 *db = nullptr;
        if (!openConnection(db)) {
            return -1;
        }

        char *errMsg = nullptr;
        int rc;
        int64_t last_id = -1;

        // Begin transaction
        rc = sqlite3_exec(db, "BEGIN TRANSACTION", 0, 0, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "Begin transaction failed: " << errMsg << std::endl;
            sqlite3_free(errMsg);
            closeConnection(db);
            return -1;
        }

        // Prepare the insert statement
        sqlite3_stmt *stmt;
        const char *sql = "INSERT INTO embeddings (chunk_text, embedding_hash, embedding_data) VALUES (?, ?, ?)";
        rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);

        if (rc != SQLITE_OK) {
            std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
            closeConnection(db);
            return -1;
        }

        // Insert each chunk and its embedding
        for (size_t i = 0; i < chunks.size(); ++i) {
            const auto &chunk = chunks[i];
            size_t hash = i < embedding_hashes.size() ? embedding_hashes[i] : 0;

            sqlite3_bind_text(stmt, 1, chunk.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int64(stmt, 2, static_cast<sqlite3_int64>(hash));
            sqlite3_bind_text(stmt, 3, embeddings_response.dump().c_str(), -1, SQLITE_STATIC);

            rc = sqlite3_step(stmt);
            if (rc != SQLITE_DONE) {
                std::cerr << "Insert failed: " << sqlite3_errmsg(db) << std::endl;
                sqlite3_finalize(stmt);
                sqlite3_exec(db, "ROLLBACK", 0, 0, 0);
                closeConnection(db);
                return -1;
            }
            sqlite3_reset(stmt);
        }

        sqlite3_finalize(stmt);

        // Commit transaction
        rc = sqlite3_exec(db, "COMMIT", 0, 0, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "Commit failed: " << errMsg << std::endl;
            sqlite3_free(errMsg);
            closeConnection(db);
            return -1;
        }

        // Get the last inserted row id
        last_id = sqlite3_last_insert_rowid(db);
        closeConnection(db);
        return last_id;
    }

    bool SQLiteDatabase::getEmbeddings(int64_t id, std::vector<std::string> &chunks, json &embeddings) {
        sqlite3 *db = nullptr;
        if (!openConnection(db)) {
            return false;
        }

        sqlite3_stmt *stmt;
        const char *sql = "SELECT chunk_text, embedding_data FROM embeddings WHERE id = ?";

        int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
        if (rc != SQLITE_OK) {
            std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
            closeConnection(db);
            return false;
        }

        sqlite3_bind_int64(stmt, 1, id);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *chunk_text = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
            const char *embedding_data = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));

            chunks.push_back(chunk_text);
            embeddings = json::parse(embedding_data);

            sqlite3_finalize(stmt);
            closeConnection(db);
            return true;
        }

        sqlite3_finalize(stmt);
        closeConnection(db);
        return false;
    }

    std::vector<std::pair<std::string, float>> SQLiteDatabase::searchSimilarVectors(const std::vector<float>& query_vector, int k) {
        std::vector<std::pair<std::string, float>> results;
        
        try {
            // Convert vector to string representation
            std::stringstream ss;
            for (size_t i = 0; i < query_vector.size(); ++i) {
                if (i > 0) ss << ",";
                ss << query_vector[i];
            }
            std::string vector_str = ss.str();

            // Create SQLite database object
            SQLite::Database db(db_path_, SQLite::OPEN_READWRITE);

            // Prepare and execute query
            std::string query = "SELECT chunk_text, similarity FROM embeddings "
                              "CROSS JOIN (SELECT embedding_data, "
                              "cosine_similarity(embedding_data, ?) as similarity "
                              "FROM embeddings ORDER BY similarity DESC LIMIT ?) AS similar_chunks "
                              "WHERE embeddings.embedding_data = similar_chunks.embedding_data";

            SQLite::Statement stmt(db, query);
            stmt.bind(1, vector_str);
            stmt.bind(2, k);

            // Process results
            while (stmt.executeStep()) {
                std::string chunk_text = stmt.getColumn(0).getString();
                float similarity = stmt.getColumn(1).getDouble();
                results.emplace_back(chunk_text, similarity);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in searchSimilarVectors: " << e.what() << std::endl;
        }

        return results;
    }

    // Get text chunks by their hash values (for NPU-accelerated similarity search)
    std::map<uint64_t, std::string> SQLiteDatabase::getChunksByHashes(const std::vector<uint64_t>& hashes) {
        std::map<uint64_t, std::string> results;
        
        if (hashes.empty()) {
            return results;
        }
        
        try {
            sqlite3* db = nullptr;
            if (!openConnection(db)) {
                return results;
            }
            
            // Prepare the query with placeholders for hash values
            std::string query = "SELECT embedding_hash, chunk_text FROM embeddings WHERE embedding_hash IN (";
            
            // Add placeholders for each hash
            for (size_t i = 0; i < hashes.size(); ++i) {
                if (i > 0) query += ",";
                query += "?";
            }
            query += ")";
            
            sqlite3_stmt *stmt;
            const char *sql = query.c_str();
            int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
            if (rc != SQLITE_OK) {
                std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
                closeConnection(db);
                return results;
            }
            
            // Bind hash values to the query
            for (size_t i = 0; i < hashes.size(); ++i) {
                sqlite3_bind_int64(stmt, i+1, static_cast<sqlite3_int64>(hashes[i])); // SQLite uses 1-based indexing for parameters
            }
            
            // Execute and collect results
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                uint64_t hash = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
                const char *text = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
                results[hash] = text;
            }
            
            sqlite3_finalize(stmt);
            closeConnection(db);
            std::cout << "Retrieved " << results.size() << " text chunks by hash from SQLite database" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error in getChunksByHashes: " << e.what() << std::endl;
        }
        
        return results;
    }
}
