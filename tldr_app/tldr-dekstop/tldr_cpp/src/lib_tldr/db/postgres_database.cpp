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
              DB_CONN_POOL_SIZE,
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

    pqxx::connection* PostgresDatabase::acquireConnection() {
        try {
            return conn_pool.acquire();
        } catch (const std::exception &e) {
            std::cerr << "Failed to acquire connection: " << e.what() << std::endl;
            return nullptr;
        }
    }

    void PostgresDatabase::releaseConnection(pqxx::connection* conn) {
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
            txn.exec("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"");

            // Create documents table
            txn.exec(
                "CREATE TABLE IF NOT EXISTS documents ("
                "id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),"
                "file_hash TEXT NOT NULL UNIQUE,"
                "file_path TEXT NOT NULL,"
                "file_name TEXT NOT NULL,"
                "title TEXT,"
                "author TEXT,"
                "subject TEXT,"
                "keywords TEXT,"
                "creator TEXT,"
                "producer TEXT,"
                "page_count INTEGER,"
                "created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,"
                "updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"
                ")"
            );


            // Create embeddings table with vector column and foreign key to documents
            txn.exec(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "id BIGSERIAL PRIMARY KEY,"
                "document_id UUID REFERENCES documents(id) ON DELETE CASCADE,"
                "chunk_text TEXT NOT NULL,"
                // "text_hash TEXT," // Store as TEXT to handle large uint64_t values
                "embedding_hash TEXT," // Store as TEXT to avoid sign issues with uint64_t
                "embedding vector(" EMBEDDING_SIZE ") NOT NULL,"  // Assuming 2048-dimensional embeddings
                "created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"
                ")"
            );

            // Create indexes for documents table
            txn.exec("CREATE INDEX IF NOT EXISTS documents_file_hash_idx ON documents (file_hash)");
            txn.exec("CREATE INDEX IF NOT EXISTS documents_created_at_idx ON documents (created_at)");

            // Create indexes for embeddings table
            // txn.exec("CREATE UNIQUE INDEX IF NOT EXISTS embeddings_text_hash_idx ON embeddings (text_hash)");
            txn.exec("CREATE UNIQUE INDEX IF NOT EXISTS embeddings_hash_idx ON embeddings (embedding_hash)");
            txn.exec("CREATE INDEX IF NOT EXISTS embeddings_document_id_idx ON embeddings (document_id)");

            // Create index for vector similarity search
            txn.exec("CREATE INDEX IF NOT EXISTS embeddings_vector_idx ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)");

            // Create a function to update the updated_at column
            txn.exec(
                "CREATE OR REPLACE FUNCTION update_updated_at_column()\n"
                "RETURNS TRIGGER AS $$\n"
                "BEGIN\n"
                "    NEW.updated_at = NOW();\n"
                "    RETURN NEW;\n"
                "END;\n"
                "$$ language 'plpgsql'"
            );

            // Create a trigger to update the updated_at column on documents
            txn.exec(
                "DO $$\n"
                "BEGIN\n"
                "    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_documents_updated_at') THEN\n"
                "        CREATE TRIGGER update_documents_updated_at\n"
                "        BEFORE UPDATE ON documents\n"
                "        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();\n"
                "    END IF;\n"
                "END\n"
                "$$;"
            );

            txn.commit();
            closeConnection(conn);
            return true;
        } catch (const std::exception &e) {
            std::cerr << "Initialization error: " << e.what() << std::endl;
            closeConnection(conn);
            return false;
        }
    }

    int64_t PostgresDatabase::saveEmbeddings(
        const std::vector<std::string_view> &chunks,
        const json &embeddings_response,
        const std::vector<uint64_t> &embedding_hashes,
        const std::vector<int> &chunk_page_nums,
        const std::string &file_hash) {

        pqxx::connection *conn = nullptr;
        if (!openConnection(conn)) {
            return -1;
        }

        // Use the connection and make sure it's released when done
        int64_t result = saveEmbeddingsWithConnection(
            conn, chunks, embeddings_response, embedding_hashes, chunk_page_nums, file_hash);
        
        // Release the connection back to the pool
        closeConnection(conn);
        
        return result;
    }

    int64_t PostgresDatabase::saveEmbeddingsWithConnection(
        pqxx::connection* conn,
        const std::vector<std::string_view> &chunks,
        const json &embeddings_response,
        const std::vector<uint64_t> &embedding_hashes,
        const std::vector<int> &chunk_page_nums,
        const std::string &file_hash) {
        
        if (!conn) {
            std::cerr << "Error: Null connection provided to saveEmbeddingsWithConnection" << std::endl;
            return -1;
        }

        try {
            pqxx::work txn(*conn);
            int64_t last_id = -1;

            // Prepare the insert statement with a unique name to avoid conflicts in multithreaded environment
            static std::atomic<int> statement_counter{0};
            std::string stmt_name = "insert_embedding_" + std::to_string(statement_counter++);

            // First, get the document_id from the documents table
            pqxx::result doc_result = txn.exec(
                "SELECT id FROM documents WHERE file_hash = $1",
                pqxx::params{file_hash}
            );
            
            if (doc_result.empty()) {
                throw std::runtime_error("Document with hash " + file_hash + " not found in database");
            }
            
            std::string document_id = doc_result[0][0].as<std::string>();
            
            // Prepare statement with updated column names and document_id
            conn->prepare(
                stmt_name,
                "INSERT INTO embeddings (document_id, chunk_text, embedding, embedding_hash) "
                "VALUES ($1, $2, $3, $4) RETURNING id"
            );

            for (size_t i = 0; i < chunks.size(); ++i) {
                // Convert embedding to string
                std::string vector_str = "[";
                const auto &embedding = embeddings_response["embeddings"][i];
                for (size_t j = 0; j < embedding.size(); ++j) {
                    if (j > 0) vector_str += ",";
                    vector_str += std::to_string(embedding[j].get<float>());
                }
                vector_str += "]";

                // Prepare parameters
                std::string chunk_str(chunks[i]);
                std::string hash_str = std::to_string(embedding_hashes[i]);
                int page_num = i < chunk_page_nums.size() ? chunk_page_nums[i] : 0;
                
                // Create params object for the prepared statement
                pqxx::params params;
                params.append(document_id);
                params.append(chunk_str);
                params.append(vector_str);
                params.append(hash_str);                             // embedding_hash as TEXT
                
                // Execute the prepared statement with the new parameter order
                auto result = txn.exec_prepared(stmt_name, params);

                // Get the last inserted id
                if (!result.empty()) {
                    last_id = result[0][0].as<int64_t>();
                }
            }
        
            txn.commit();
            return last_id;
        } catch (const std::exception &e) {
            std::cerr << "Insertion error in saveEmbeddingsWithConnection: " << e.what() << std::endl;
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

            // Use the newer exec method directly with the params vector
            auto result = txn.exec(
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

    std::vector<std::tuple<std::string, float, uint64_t>> PostgresDatabase::searchSimilarVectors(const std::vector<float>& query_vector, int k) {
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
                "SELECT chunk_text, 1 - (embedding <=> " + vector_str + ") as similarity, embedding_hash "
                "FROM embeddings "
                "ORDER BY embedding <=> " + vector_str + " "
                "LIMIT " + std::to_string(k);

            auto result = txn.exec(query);
            std::vector<std::tuple<std::string, float, uint64_t>> results;

            for (const auto& row : result) {
                // Convert the hash from string to uint64_t
                std::string hash_str = row["embedding_hash"].as<std::string>();
                uint64_t hash = std::stoull(hash_str);
                
                results.emplace_back(
                    row["chunk_text"].as<std::string>(),
                    row["similarity"].as<float>(),
                    hash
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

    // Get text chunks by their hash values
    std::map<uint64_t, std::string> PostgresDatabase::getChunksByHashes(const std::vector<uint64_t>& hashes) {
        std::map<uint64_t, std::string> results;
        
        if (hashes.empty()) {
            return results;
        }
        
        pqxx::connection *conn = nullptr;
        if (!openConnection(conn)) {
            return results;
        }
        
        try {
            pqxx::work txn(*conn);
            
            // Build the query with parameter placeholders
            // Note: both text_hash and embedding_hash are now TEXT in the database
            std::string query = "SELECT embedding_hash, chunk_text FROM embeddings WHERE embedding_hash IN (";
            
            // Create parameters list and placeholder string
            std::cout << "hashes for search_query:" << std::endl;
            for (size_t i = 0; i < hashes.size(); ++i) {
                if (i > 0) query += ",";
                std::string hash_str = std::to_string(hashes[i]);
                query += "'" + hash_str + "'";
                std::cout << hash_str << "\n";
            }
            query += ")";

            // Use the newer exec method directly with the params vector
            pqxx::result db_result = txn.exec(query);
            
            // Process results
            for (const auto& row : db_result) {
                // Convert the hash from string to uint64_t
                std::string hash_str = row["embedding_hash"].as<std::string>();
                uint64_t hash = std::stoull(hash_str);
                std::string text = row["chunk_text"].as<std::string>();
                results[hash] = text;
            }
            
            txn.commit();
            std::cout << "Retrieved " << results.size() << " text chunks by hash from PostgreSQL database" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error in getChunksByHashes: " << e.what() << std::endl;
        }
        
        closeConnection(conn);
        return results;
    }

    bool PostgresDatabase::saveDocumentMetadata(
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
        
        // Validate input parameters
        if (fileHash.empty()) {
            std::cerr << "Error in saveDocumentMetadata: fileHash is empty" << std::endl;
            return false;
        }
        
        if (filePath.empty()) {
            std::cerr << "Error in saveDocumentMetadata: filePath is empty" << std::endl;
            return false;
        }
        
        if (fileName.empty()) {
            std::cerr << "Error in saveDocumentMetadata: fileName is empty" << std::endl;
            return false;
        }
        
        pqxx::connection *conn = nullptr;
        if (!openConnection(conn)) {
            return false;
        }

        try {
            pqxx::work txn(*conn);
            
            // First check if document with this hash already exists
            auto result = txn.exec(
                "SELECT 1 FROM documents WHERE file_hash = $1",
                pqxx::params{fileHash}
            );

            if (result.empty()) {
                // Insert new document - create a copy of all parameters to ensure they stay in scope
                std::string fileHashCopy = fileHash;
                std::string filePathCopy = filePath;
                std::string fileNameCopy = fileName;
                std::string titleCopy = title.empty() ? "" : title;
                std::string authorCopy = author.empty() ? "" : author;
                std::string subjectCopy = subject.empty() ? "" : subject;
                std::string keywordsCopy = keywords.empty() ? "" : keywords;
                std::string creatorCopy = creator.empty() ? "" : creator;
                std::string producerCopy = producer.empty() ? "" : producer;
                int pageCountCopy = pageCount;
                
                // Create params object with copies of all parameters
                pqxx::params insertParams;
                insertParams.append(fileHashCopy);
                insertParams.append(filePathCopy);
                insertParams.append(fileNameCopy);
                
                // Handle optional string fields safely
                if (titleCopy.empty()) {
                    insertParams.append(nullptr);
                } else {
                    insertParams.append(titleCopy);
                }
                
                if (authorCopy.empty()) {
                    insertParams.append(nullptr);
                } else {
                    insertParams.append(authorCopy);
                }
                
                if (subjectCopy.empty()) {
                    insertParams.append(nullptr);
                } else {
                    insertParams.append(subjectCopy);
                }
                
                if (keywordsCopy.empty()) {
                    insertParams.append(nullptr);
                } else {
                    insertParams.append(keywordsCopy);
                }
                
                if (creatorCopy.empty()) {
                    insertParams.append(nullptr);
                } else {
                    insertParams.append(creatorCopy);
                }
                
                if (producerCopy.empty()) {
                    insertParams.append(nullptr);
                } else {
                    insertParams.append(producerCopy);
                }
                
                insertParams.append(pageCountCopy);
                
                // Execute the query with the params object
                txn.exec(
                    "INSERT INTO documents (file_hash, file_path, file_name, title, author, "
                    "subject, keywords, creator, producer, page_count, created_at, updated_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                    insertParams
                );
            } else {
                // Update existing document - create a copy of all parameters to ensure they stay in scope
                std::string fileHashCopy = fileHash;
                std::string filePathCopy = filePath;
                std::string fileNameCopy = fileName;
                std::string titleCopy = title.empty() ? "" : title;
                std::string authorCopy = author.empty() ? "" : author;
                std::string subjectCopy = subject.empty() ? "" : subject;
                std::string keywordsCopy = keywords.empty() ? "" : keywords;
                std::string creatorCopy = creator.empty() ? "" : creator;
                std::string producerCopy = producer.empty() ? "" : producer;
                int pageCountCopy = pageCount;
                
                // Create params object with copies of all parameters
                pqxx::params updateParams;
                updateParams.append(filePathCopy);
                updateParams.append(fileNameCopy);
                
                // Handle optional string fields safely
                if (titleCopy.empty()) {
                    updateParams.append(nullptr);
                } else {
                    updateParams.append(titleCopy);
                }
                
                if (authorCopy.empty()) {
                    updateParams.append(nullptr);
                } else {
                    updateParams.append(authorCopy);
                }
                
                if (subjectCopy.empty()) {
                    updateParams.append(nullptr);
                } else {
                    updateParams.append(subjectCopy);
                }
                
                if (keywordsCopy.empty()) {
                    updateParams.append(nullptr);
                } else {
                    updateParams.append(keywordsCopy);
                }
                
                if (creatorCopy.empty()) {
                    updateParams.append(nullptr);
                } else {
                    updateParams.append(creatorCopy);
                }
                
                if (producerCopy.empty()) {
                    updateParams.append(nullptr);
                } else {
                    updateParams.append(producerCopy);
                }
                
                updateParams.append(pageCountCopy);
                updateParams.append(fileHashCopy);
                
                // Execute the query with the params object
                txn.exec(
                    "UPDATE documents SET "
                    "file_path = $1, "
                    "file_name = $2, "
                    "title = $3, "
                    "author = $4, "
                    "subject = $5, "
                    "keywords = $6, "
                    "creator = $7, "
                    "producer = $8, "
                    "page_count = $9, "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE file_hash = $10",
                    updateParams
                );
            }
            
            txn.commit();
            closeConnection(conn);
            return true;
        } catch (const std::exception &e) {
            std::cerr << "Error in saveDocumentMetadata: " << e.what() << std::endl;
            closeConnection(conn);
            return false;
        }
    }
}

bool tldr::PostgresDatabase::deleteEmbeddings(const std::string& file_hash) {
    if (file_hash.empty()) {
        std::cerr << "Cannot delete embeddings: empty file hash provided" << std::endl;
        return false;
    }

    pqxx::connection* conn = nullptr;
    if (!openConnection(conn)) {
        return false;
    }

    try {
        pqxx::work txn(*conn);
        
        // First get the document_id for the file_hash
        pqxx::result doc_result = txn.exec(
            "SELECT id FROM documents WHERE file_hash = $1",
            pqxx::params{file_hash}
        );
        
        if (doc_result.empty()) {
            std::cerr << "No document found with hash: " << file_hash << std::endl;
            closeConnection(conn);
            return false;
        }
        
        std::string document_id = doc_result[0][0].as<std::string>();
        
        // Delete all embeddings for the given document_id
        std::string delete_sql = "DELETE FROM embeddings WHERE document_id = $1";
        auto result = txn.exec(delete_sql, pqxx::params{document_id});
        
        txn.commit();
        closeConnection(conn);
        
        std::cout << "Deleted " << result.affected_rows() << " embeddings for file hash: " << file_hash << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error deleting embeddings: " << e.what() << std::endl;
        if (conn) {
            closeConnection(conn);
        }
        return false;
    }
}
