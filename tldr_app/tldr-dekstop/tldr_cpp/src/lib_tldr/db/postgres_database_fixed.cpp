#include "postgres_database.h"
#include <iostream>
#include <atomic>
#include <pqxx/pqxx>
#include "../constants.h"

namespace tldr {

    // Implementation of saveDocumentMetadata function
    bool PostgresDatabase::saveDocumentMetadata(
        const std::string &fileHash, const std::string &filePath, const std::string &fileName,
        const std::string &title, const std::string &author, const std::string &subject,
        const std::string &keywords, const std::string &creator, const std::string &producer, int pageCount) {
        
        // Validate required input parameters
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

        // Open database connection
        pqxx::connection *conn = nullptr;
        if (!openConnection(conn)) {
            return false;
        }
        
        // Guard against null connection
        if (conn == nullptr) {
            std::cerr << "Error in saveDocumentMetadata: null connection" << std::endl;
            return false;
        }

        try {
            pqxx::work txn(*conn);

            // Check if document with this hash already exists
            pqxx::result result = txn.exec_params(
                "SELECT 1 FROM documents WHERE file_hash = $1",
                fileHash
            );

            if (result.empty()) {
                // Document doesn't exist - insert new record
                std::string query = 
                    "INSERT INTO documents "
                    "(file_hash, file_path, file_name, title, author, subject, keywords, creator, producer, page_count, created_at, updated_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)";

                // Use direct parameter passing to avoid potential issues with pqxx::params
                txn.exec_params(query,
                    fileHash,
                    filePath,
                    fileName,
                    title.empty() ? "" : title,
                    author.empty() ? "" : author,
                    subject.empty() ? "" : subject,
                    keywords.empty() ? "" : keywords,
                    creator.empty() ? "" : creator,
                    producer.empty() ? "" : producer,
                    pageCount
                );
            } else {
                // Document exists - update record
                std::string query = 
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
                    "WHERE file_hash = $10";

                // Use direct parameter passing
                txn.exec_params(query,
                    filePath,
                    fileName,
                    title.empty() ? "" : title,
                    author.empty() ? "" : author,
                    subject.empty() ? "" : subject,
                    keywords.empty() ? "" : keywords,
                    creator.empty() ? "" : creator,
                    producer.empty() ? "" : producer,
                    pageCount,
                    fileHash
                );
            }

            txn.commit();
            
            // Make sure connection is valid before closing
            if (conn) {
                closeConnection(conn);
            }
            return true;
        } catch (const std::exception &e) {
            std::cerr << "Error in saveDocumentMetadata: " << e.what() << std::endl;
            // Make sure connection is valid before closing
            if (conn) {
                closeConnection(conn);
            }
            return false;
        }
    }
}
