#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <sqlite3.h>
#include <openssl/sha.h>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

// Using nlohmann/json for JSON parsing
using json = nlohmann::json;

namespace fs = std::filesystem;

struct Document {
    std::string doc_path;
    std::string doc_title;
    std::string shasum;
};

// Function to read JSON and return vector of Documents
std::vector<Document> readDocumentsFromJson(const std::string& filename) {
    std::vector<Document> documents;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return documents;
    }

    try {
        json j;
        file >> j;

        for (const auto& item : j) {
            Document doc;
            doc.doc_path = item["doc_path"];
            doc.doc_title = item["doc_title"];
            doc.shasum = item["shasum"];
            documents.push_back(doc);
        }
    } catch (json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
    }

    return documents;
}


class PdfTracker {
private:
    sqlite3* db;



    void initDatabase() {
        const char* sql =
            "CREATE TABLE IF NOT EXISTS corpus_files ("
            "path TEXT PRIMARY KEY,"
            "title TEXT,"
            "shasum TEXT,"
            "last_seen INTEGER"
            ");";

        char* errMsg = nullptr;
        if (sqlite3_exec(db, sql, nullptr, nullptr, &errMsg) != SQLITE_OK) {
            std::cerr << "SQL error: " << errMsg << std::endl;
            sqlite3_free(errMsg);
        }
    }

public:
    PdfTracker(const std::string& dbPath) {
        if (sqlite3_open(dbPath.c_str(), &db) != SQLITE_OK) {
            throw std::runtime_error("Cannot open database");
        }
        initDatabase();
    }

    ~PdfTracker() {
        sqlite3_close(db);
    }

    std::vector<PdfInfo> scanDirectory(const std::string& dirPath) {
        std::vector<PdfInfo> pdfs;

        for(const auto& entry : fs::recursive_directory_iterator(dirPath)) {
            if(entry.path().extension() == ".pdf") {
                PdfInfo info;
                info.path = fs::absolute(entry.path()).string();
                info.name = entry.path().filename().string();
                info.title = getDocumentTitle(info.path);
                info.shasum = calculateSHA256(info.path);
                pdfs.push_back(info);
            }
        }

        return pdfs;
    }

    void updateDatabase(const std::vector<PdfInfo>& currentFiles) {
        // Mark all files as not seen
        const char* markUnseen = "UPDATE corpus_files SET last_seen = 0;";
        sqlite3_exec(db, markUnseen, nullptr, nullptr, nullptr);

        for(const auto& file : currentFiles) {
            // Check if file exists in database
            sqlite3_stmt* stmt;
            const char* sql = "SELECT shasum FROM corpus_files WHERE path = ?;";
            sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
            sqlite3_bind_text(stmt, 1, file.path.c_str(), -1, SQLITE_STATIC);

            if(sqlite3_step(stmt) == SQLITE_ROW) {
                // File exists, check if modified
                std::string oldShasum = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
                if(oldShasum != file.shasum) {
                    std::cout << "Updated: " << file.path << std::endl;
                }
                // Update record
                const char* updateSql =
                    "UPDATE corpus_files SET name = ?, title = ?, shasum = ?, last_seen = 1 "
                    "WHERE path = ?;";
                sqlite3_stmt* updateStmt;
                sqlite3_prepare_v2(db, updateSql, -1, &updateStmt, nullptr);
                sqlite3_bind_text(updateStmt, 1, file.name.c_str(), -1, SQLITE_STATIC);
                sqlite3_bind_text(updateStmt, 2, file.title.c_str(), -1, SQLITE_STATIC);
                sqlite3_bind_text(updateStmt, 3, file.shasum.c_str(), -1, SQLITE_STATIC);
                sqlite3_bind_text(updateStmt, 4, file.path.c_str(), -1, SQLITE_STATIC);
                sqlite3_step(updateStmt);
                sqlite3_finalize(updateStmt);
            } else {
                // New file
                std::cout << "New: " << file.path << std::endl;
                const char* insertSql =
                    "INSERT INTO corpus_files (path, name, title, shasum, last_seen) "
                    "VALUES (?, ?, ?, ?, 1);";
                sqlite3_stmt* insertStmt;
                sqlite3_prepare_v2(db, insertSql, -1, &insertStmt, nullptr);
                sqlite3_bind_text(insertStmt, 1, file.path.c_str(), -1, SQLITE_STATIC);
                sqlite3_bind_text(insertStmt, 2, file.name.c_str(), -1, SQLITE_STATIC);
                sqlite3_bind_text(insertStmt, 3, file.title.c_str(), -1, SQLITE_STATIC);
                sqlite3_bind_text(insertStmt, 4, file.shasum.c_str(), -1, SQLITE_STATIC);
                sqlite3_step(insertStmt);
                sqlite3_finalize(insertStmt);
            }
            sqlite3_finalize(stmt);
        }

        // Find deleted files
        const char* findDeleted = "SELECT path FROM corpus_files WHERE last_seen = 0;";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, findDeleted, -1, &stmt, nullptr);
        while(sqlite3_step(stmt) == SQLITE_ROW) {
            std::string deletedPath = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            std::cout << "Deleted: " << deletedPath << std::endl;
        }
        sqlite3_finalize(stmt);

        // Clean up deleted files from database
        const char* cleanup = "DELETE FROM corpus_files WHERE last_seen = 0;";
        sqlite3_exec(db, cleanup, nullptr, nullptr, nullptr);
    }
};

void read_doc_statuses() {
    std::string filename = "documents.json";
    std::vector<Document> docs = readDocumentsFromJson(filename);

    // Print the loaded documents
    for (const auto& doc : docs) {
        std::cout << "Path: " << doc.doc_path << std::endl;
        std::cout << "Title: " << doc.doc_title << std::endl;
        std::cout << "Shasum: " << doc.shasum << std::endl;
        std::cout << "-------------------" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <directory_path> <database_path>" << std::endl;
        return 1;
    }

    try {
        PdfTracker tracker(argv[2]);
        std::vector<PdfInfo> currentFiles = tracker.scanDirectory(argv[1]);
        tracker.updateDatabase(currentFiles);
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
