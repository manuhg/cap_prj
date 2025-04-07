#include "sqlite_database.h"
#include <iostream>

namespace tldr {
    SQLiteDatabase::SQLiteDatabase(const std::string& db_path) 
        : db_path_(db_path), db_(nullptr) {}

    SQLiteDatabase::~SQLiteDatabase() {
        closeConnection();
    }

    bool SQLiteDatabase::openConnection() {
        if (db_ != nullptr) {
            return true;
        }

        int rc = sqlite3_open(db_path_.c_str(), &db_);
        if (rc) {
            std::cerr << "Can't open database: " << sqlite3_errmsg(db_) << std::endl;
            return false;
        }
        return true;
    }

    void SQLiteDatabase::closeConnection() {
        if (db_ != nullptr) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
    }

    bool SQLiteDatabase::initialize() {
        if (!openConnection()) {
            return false;
        }

        char* errMsg = nullptr;
        int rc;

        // Enable WAL mode for better performance and concurrency
        rc = sqlite3_exec(db_, "PRAGMA journal_mode=WAL", 0, 0, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "Failed to enable WAL mode: " << errMsg << std::endl;
            sqlite3_free(errMsg);
            return false;
        }

        // Set other recommended WAL-related PRAGMAs
        rc = sqlite3_exec(db_, "PRAGMA synchronous=NORMAL", 0, 0, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "Failed to set synchronous mode: " << errMsg << std::endl;
            sqlite3_free(errMsg);
            return false;
        }

        const char* sql = "CREATE TABLE IF NOT EXISTS embeddings ("
                         "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                         "chunk_text TEXT NOT NULL,"
                         "embedding_data TEXT NOT NULL,"
                         "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)";

        rc = sqlite3_exec(db_, sql, 0, 0, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "SQL error: " << errMsg << std::endl;
            sqlite3_free(errMsg);
            return false;
        }

        return true;
    }

    int64_t SQLiteDatabase::saveEmbeddings(const std::vector<std::string>& chunks, const json& embeddings_response) {
        if (!openConnection()) {
            return -1;
        }

        char* errMsg = nullptr;
        int rc;
        int64_t last_id = -1;

        // Begin transaction
        rc = sqlite3_exec(db_, "BEGIN TRANSACTION", 0, 0, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "Begin transaction failed: " << errMsg << std::endl;
            sqlite3_free(errMsg);
            return -1;
        }

        // Prepare the insert statement
        sqlite3_stmt* stmt;
        const char* sql = "INSERT INTO embeddings (chunk_text, embedding_data) VALUES (?, ?)";
        rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, 0);

        if (rc != SQLITE_OK) {
            std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db_) << std::endl;
            return -1;
        }

        // Insert each chunk and its embedding
        for (const auto& chunk : chunks) {
            sqlite3_bind_text(stmt, 1, chunk.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 2, embeddings_response.dump().c_str(), -1, SQLITE_STATIC);

            rc = sqlite3_step(stmt);
            if (rc != SQLITE_DONE) {
                std::cerr << "Insert failed: " << sqlite3_errmsg(db_) << std::endl;
                sqlite3_finalize(stmt);
                sqlite3_exec(db_, "ROLLBACK", 0, 0, 0);
                return -1;
            }
            sqlite3_reset(stmt);
        }

        sqlite3_finalize(stmt);

        // Commit transaction
        rc = sqlite3_exec(db_, "COMMIT", 0, 0, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "Commit failed: " << errMsg << std::endl;
            sqlite3_free(errMsg);
            return -1;
        }

        // Get the last inserted row id
        last_id = sqlite3_last_insert_rowid(db_);
        return last_id;
    }

    bool SQLiteDatabase::getEmbeddings(int64_t id, std::vector<std::string>& chunks, json& embeddings) {
        if (!openConnection()) {
            return false;
        }

        sqlite3_stmt* stmt;
        const char* sql = "SELECT chunk_text, embedding_data FROM embeddings WHERE id = ?";
        
        int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, 0);
        if (rc != SQLITE_OK) {
            std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db_) << std::endl;
            return false;
        }

        sqlite3_bind_int64(stmt, 1, id);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* chunk_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            const char* embedding_data = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            
            chunks.push_back(chunk_text);
            embeddings = json::parse(embedding_data);
            
            sqlite3_finalize(stmt);
            return true;
        }

        sqlite3_finalize(stmt);
        return false;
    }
}
