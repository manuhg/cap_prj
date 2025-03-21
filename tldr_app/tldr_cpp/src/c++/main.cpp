#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <pqxx/pqxx>
#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-page.h>

#define PAGE_DELIMITER "\n\n"

// Simple function to extract text from a PDF file
std::string extractTextFromPDF(const std::string &filename) {
    // Load the PDF document
    poppler::document *doc = poppler::document::load_from_file(filename);

    // Check if the document loaded successfully
    if (!doc) {
        std::cerr << "Error opening PDF file." << std::endl;
        return "";
    }

    std::string text;

    // Iterate through all pages
    for (int i = 0; i < doc->pages(); ++i) {
        // Get the current page
        poppler::page *page = doc->create_page(i);
        if (page) {
            // Extract text from the page and append it to our result
            poppler::byte_array utf8_data = page->text().to_utf8();
            std::string page_text(utf8_data.begin(), utf8_data.end());

            text.append(page_text + PAGE_DELIMITER);
            delete page;
        }
    }

    // Clean up
    delete doc;

    return text;
}

void doRag(const std::string &conversationId) {
    // Implement the function
    std::cout << "DO_RAG action with conversation_id: " << conversationId << std::endl;

    try {
        pqxx::connection C("dbname=testdb user=postgres password=secret hostaddr=127.0.0.1 port=5432");
        pqxx::work W(C);

        std::string query =
                "SELECT * FROM conversations WHERE id = " + W.quote(conversationId) + " ORDER BY created_at";
        pqxx::result R = W.exec(query);

        for (auto row: R) {
            std::cout << "ID: " << row["id"].c_str() << ", Created At: " << row["created_at"].c_str() << std::endl;
        }

        W.commit();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}

void addCorpus(const std::string &sourcePath) {
    // Implement the function
    std::cout << "ADD_CORPUS action with source_path: " << sourcePath << std::endl;
}

void deleteCorpus(const std::string &corpusId) {
    // Implement the function
    std::cout << "DELETE_CORPUS action with corpus_id: " << corpusId << std::endl;
}

int main() {
    std::string input;
    std::map<std::string, std::function<void(const std::string &)>> actions = {
            {"do-rag",        doRag},
            {"add-corpus",    addCorpus},
            {"delete-corpus", deleteCorpus}
    };

    while (true) {
        std::cout << "Enter command: ";
        std::getline(std::cin, input);
        std::istringstream iss(input);
        std::string command, argument;
        iss >> command >> argument;

        auto it = actions.find(command);
        if (it != actions.end()) {
            it->second(argument);
        } else if (command == "exit") {
            break;
        } else {
            std::cout << "Unknown command: " << command << std::endl;
        }
    }
    return 0;
}