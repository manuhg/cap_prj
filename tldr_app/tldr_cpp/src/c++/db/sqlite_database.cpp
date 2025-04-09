#include "sqlite_database.h"
#include <iostream>

namespace tldr {

    SQLiteDatabase::SQLiteDatabase(const std::string& db_path) 
        : db_path_(db_path), 
          conn_pool(
              db_path, 
              CONN_POOL_SIZE,
              [](const std::string& path) { 
                  sqlite3* db = nullptr;
                  int rc = sqlite3_open(path.c_str(), &db);
                  if (rc) {
                      throw std::runtime_error("Can't open database: " + std::string(sqlite3_errmsg(db)));
                  }
                  return db;
              }, 
              [](sqlite3* db) {
                  if (db) {
                      sqlite3_close(db);
                  }
              }
          ) {}

    SQLiteDatabase::~SQLiteDatabase() {}

    bool SQLiteDatabase::openConnection(sqlite3*& db) {
        try {
            db = conn_pool.acquire();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to acquire connection: " << e.what() << std::endl;
            return false;
        }
    }

    void SQLiteDatabase::closeConnection(sqlite3* db) {
        if (db) {
            conn_pool.release(db);
        }
    }

    bool SQLiteDatabase::initialize() {
        sqlite3* db = nullptr;
        if (!openConnection(db)) {
            return false;
        }

        char* errMsg = nullptr;
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

        const char* sql = "CREATE TABLE IF NOT EXISTS embeddings ("
                         "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                         "chunk_text TEXT NOT NULL,"
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

    int64_t SQLiteDatabase::saveEmbeddings(const std::vector<std::string>& chunks, const json& embeddings_response) {
        sqlite3* db = nullptr;
        if (!openConnection(db)) {
            return -1;
        }

        char* errMsg = nullptr;
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
        sqlite3_stmt* stmt;
        const char* sql = "INSERT INTO embeddings (chunk_text, embedding_data) VALUES (?, ?)";
        rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);

        if (rc != SQLITE_OK) {
            std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
            closeConnection(db);
            return -1;
        }

        // Insert each chunk and its embedding
        for (const auto& chunk : chunks) {
            sqlite3_bind_text(stmt, 1, chunk.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 2, embeddings_response.dump().c_str(), -1, SQLITE_STATIC);

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

    bool SQLiteDatabase::getEmbeddings(int64_t id, std::vector<std::string>& chunks, json& embeddings) {
        sqlite3* db = nullptr;
        if (!openConnection(db)) {
            return false;
        }

        sqlite3_stmt* stmt;
        const char* sql = "SELECT chunk_text, embedding_data FROM embeddings WHERE id = ?";
        
        int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
        if (rc != SQLITE_OK) {
            std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
            closeConnection(db);
            return false;
        }

        sqlite3_bind_int64(stmt, 1, id);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* chunk_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            const char* embedding_data = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            
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
}
