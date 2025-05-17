#include "sqlite_database.h"
#include <SQLiteCpp/SQLiteCpp.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace tldr {
    SQLiteDatabase::SQLiteDatabase(const std::string &db_path)
        : db_path_(db_path) {
        try {
            // Initialize the database with WAL mode for better concurrency
            SQLite::Database db(db_path, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
            db.exec("PRAGMA journal_mode=WAL");
            db.exec("PRAGMA synchronous=NORMAL");
            db.exec("PRAGMA cache_size=10000");
            db.exec("PRAGMA foreign_keys = ON");
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to initialize SQLite database: " + std::string(e.what()));
        }
    }

    SQLiteDatabase::~SQLiteDatabase() {
        // No need to manually close connections as SQLiteCpp manages them
    }

    bool SQLiteDatabase::openConnection(sqlite3*& db) {
        // This method is kept for backward compatibility but is no longer used
        // as we're now using SQLiteCpp which manages its own connections
        return true;
    }
    
    bool SQLiteDatabase::saveDocumentMetadata(
        const std::string& fileHash,
        const std::string& filePath,
        const std::string& fileName,
        const std::string& title,
        const std::string& author,
        const std::string& subject,
        const std::string& keywords,
        const std::string& creator,
        const std::string& producer,
        int pageCount) {
        
        try {
            SQLite::Database db(db_path_, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
            
            // Create documents table if it doesn't exist
            const char* createTableSQL = R"(
                CREATE TABLE IF NOT EXISTS documents (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    title TEXT,
                    author TEXT,
                    subject TEXT,
                    keywords TEXT,
                    creator TEXT,
                    producer TEXT,
                    page_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            )";
            db.exec(createTableSQL);
            
            // Insert or replace document metadata
            SQLite::Statement query(db, R"(
                INSERT OR REPLACE INTO documents (
                    file_hash, file_path, file_name, title, author, 
                    subject, keywords, creator, producer, page_count, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            )");
            
            // Bind parameters
            query.bind(1, fileHash);
            query.bind(2, filePath);
            query.bind(3, fileName);
            query.bind(4, title);
            query.bind(5, author);
            query.bind(6, subject);
            query.bind(7, keywords);
            query.bind(8, creator);
            query.bind(9, producer);
            query.bind(10, pageCount);
            
            // Execute the query
            int rowsAffected = query.exec();
            return rowsAffected > 0;
            
        } catch (const std::exception& e) {
            std::cerr << "Error saving document metadata: " << e.what() << std::endl;
            return false;
        }
    }

    void SQLiteDatabase::closeConnection(sqlite3* db) {
        // This method is kept for backward compatibility but is no longer used
        // as we're now using SQLiteCpp which manages its own connections
    }

    bool SQLiteDatabase::initialize() {
        try {
            // Open a connection to the database
            SQLite::Database db(db_path_, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
            
            // Create the embeddings table if it doesn't exist
            db.exec(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "chunk_text TEXT NOT NULL, "
                "embedding_hash TEXT, " // Store as TEXT to avoid issues with uint64_t
                "embedding_data TEXT NOT NULL, "
                "chunk_page_num INTEGER NOT NULL, "
                "file_hash TEXT NOT NULL, "
                "created_at DATETIME DEFAULT CURRENT_TIMESTAMP, "
                "UNIQUE(embedding_hash, file_hash)"
                ")");
                
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error initializing database: " << e.what() << std::endl;
            return false;
        }
    }

    int64_t SQLiteDatabase::saveEmbeddings(
        const std::vector<std::string_view> &chunks,
        const json &embeddings_response,
        const std::vector<uint64_t> &embedding_hashes,
        const std::vector<int>& chunkPageNums,
        const std::string& fileHash) {
        
        if (chunks.empty() || chunks.size() != embedding_hashes.size() || chunks.size() != chunkPageNums.size()) {
            std::cerr << "Error: Mismatch in input sizes" << std::endl;
            return -1;
        }

        try {
            // Open a connection to the database
            SQLite::Database db(db_path_, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
            
            // Create the embeddings table if it doesn't exist
            db.exec(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "chunk_text TEXT NOT NULL, "
                "embedding_hash INTEGER NOT NULL, "
                "embedding_data TEXT NOT NULL, "
                "chunk_page_num INTEGER NOT NULL, "
                "file_hash TEXT NOT NULL, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "UNIQUE(embedding_hash, file_hash)"
                ")");

            // Enable WAL mode for better concurrency
            db.exec("PRAGMA journal_mode=WAL");
            
            // Enable foreign keys
            db.exec("PRAGMA foreign_keys = ON");

            // Prepare the insert statement
            SQLite::Statement stmt(db, 
                "INSERT OR REPLACE INTO embeddings (chunk_text, embedding_hash, embedding_data, chunk_page_num, file_hash) "
                "VALUES (?, ?, ?, ?, ?)");

            // Begin transaction
            SQLite::Transaction transaction(db);
            int64_t last_id = -1;
            
            // Insert each chunk and its embedding
            for (size_t i = 0; i < chunks.size(); ++i) {
                const auto &chunk = chunks[i];
                uint64_t hash = i < embedding_hashes.size() ? embedding_hashes[i] : 0;
                int page_num = i < chunkPageNums.size() ? chunkPageNums[i] : 0;
                
                // Convert embedding to string
                std::string vector_str = "[";
                const auto &embedding = embeddings_response["embeddings"][i];
                for (size_t j = 0; j < embedding.size(); ++j) {
                    if (j > 0) vector_str += ",";
                    vector_str += std::to_string(embedding[j].get<float>());
                }
                vector_str += "]";
                
                // Bind parameters
                stmt.bind(1, std::string(chunk));
                stmt.bind(2, static_cast<int64_t>(hash));
                stmt.bind(3, vector_str);
                stmt.bind(4, page_num);
                stmt.bind(5, fileHash);
                
                // Execute the statement
                stmt.exec();
                last_id = db.getLastInsertRowid();
                
                // Reset the statement for the next iteration
                stmt.reset();
            }
            
            // Commit the transaction
            transaction.commit();
            return last_id;
        } catch (const std::exception& e) {
            std::cerr << "Error in saveEmbeddings: " << e.what() << std::endl;
            return -1;
        }    return -1;
        }
    }

    bool tldr::SQLiteDatabase::getEmbeddings(int64_t id, std::vector<std::string> &chunks, json &embeddings) {
        try {
            // Open a connection to the database
            SQLite::Database db(db_path_, SQLite::OPEN_READONLY);
            
            // Prepare the query
            SQLite::Statement query(db, 
                "SELECT chunk_text, embedding_data FROM embeddings WHERE id = ?");
            query.bind(1, id);
            
            // Execute the query
            if (query.executeStep()) {
                // Get chunk text
                std::string chunk_text = query.getColumn(0).getString();
                chunks.push_back(chunk_text);
                
                // Parse embeddings JSON
                std::string embedding_json = query.getColumn(1).getString();
                embeddings = json::parse(embedding_json);
                
                return true;
            }
            
            return false;
        } catch (const std::exception& e) {
            std::cerr << "Error in getEmbeddings: " << e.what() << std::endl;
            return false;
        }
    }

    std::vector<std::tuple<std::string, float, uint64_t>> tldr::SQLiteDatabase::searchSimilarVectors(const std::vector<float>& query_vector, int k) {
        std::vector<std::tuple<std::string, float, uint64_t>> results;
        
        try {
            // Convert query vector to string for SQL
            std::string query_vec_str = "[";
            for (size_t i = 0; i < query_vector.size(); ++i) {
                if (i > 0) query_vec_str += ",";
                query_vec_str += std::to_string(query_vector[i]);
            }
            query_vec_str += "]";

            // Open a connection to the database
            SQLite::Database db(db_path_, SQLite::OPEN_READONLY);
            
            // Prepare the similarity search query
            std::string query = 
                "SELECT chunk_text, 1 - (embedding <=> ?) as similarity, embedding_hash "
                "FROM embeddings "
                "ORDER BY embedding <=> ? "
                "LIMIT " + std::to_string(k);
                
            SQLite::Statement stmt(db, query);
            stmt.bind(1, query_vec_str);
            stmt.bind(2, query_vec_str);
            
            // Execute the query and process results
            while (stmt.executeStep()) {
                std::string chunk_text = stmt.getColumn(0).getString();
                float similarity = static_cast<float>(stmt.getColumn(1).getDouble());
                std::string hash_str = stmt.getColumn(2).getString();
                uint64_t hash = std::stoull(hash_str);
                
                results.emplace_back(chunk_text, similarity, hash);
            }
            
            return results;
        } catch (const std::exception& e) {
            std::cerr << "Error in searchSimilarVectors: " << e.what() << std::endl;
            return {};
        }
    }

    // Get text chunks by their hash values (for NPU-accelerated similarity search)
    std::map<uint64_t, std::string> tldr::SQLiteDatabase::getChunksByHashes(const std::vector<uint64_t>& hashes) {
        std::map<uint64_t, std::string> results;
        
        if (hashes.empty()) {
            return results;
        }
        
        try {
            // Open a connection to the database
            SQLite::Database db(db_path_, SQLite::OPEN_READONLY);
            
            // Build the query with parameter placeholders
            std::string query = "SELECT embedding_hash, chunk_text FROM embeddings WHERE embedding_hash IN (";
            
            // Create placeholders for each hash
            for (size_t i = 0; i < hashes.size(); ++i) {
                if (i > 0) query += ",";
                query += "?";
            }
            query += ")";
            
            // Prepare the query
            SQLite::Statement stmt(db, query);
            
            // Bind each hash value as text
            for (size_t i = 0; i < hashes.size(); ++i) {
                stmt.bind(i + 1, std::to_string(hashes[i]));
            }
            
            // Execute the query and process results
            while (stmt.executeStep()) {
                std::string hash_str = stmt.getColumn(0).getString();
                uint64_t hash = std::stoull(hash_str);
                std::string chunk_text = stmt.getColumn(1).getString();
                results[hash] = chunk_text;
            }
            
            std::cout << "Retrieved " << results.size() << " text chunks by hash from SQLite database" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error in getChunksByHashes: " << e.what() << std::endl;
        }
        
        return results;
    }


bool tldr::SQLiteDatabase::deleteEmbeddings(const std::string& file_hash) {
    if (file_hash.empty()) {
        std::cerr << "Cannot delete embeddings: empty file hash provided" << std::endl;
        return false;
    }

    try {
        // Open a connection to the database
        SQLite::Database db(db_path_, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
        
        // Delete all embeddings for the given file hash
        SQLite::Statement query(db, "DELETE FROM embeddings WHERE file_hash = ?");
        query.bind(1, file_hash);
        
        int rows_affected = query.exec();
        
        std::cout << "Deleted " << rows_affected << " embeddings for file hash: " << file_hash << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error deleting embeddings: " << e.what() << std::endl;
        return false;
    }
}
