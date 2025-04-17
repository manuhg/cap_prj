#include <iostream>
#include <string>
#include "lib_tldr/tldr_lib.h"
#include "lib_tldr/constants.h"
#include <map>
#include <sstream>
#include <functional>

void command_loop(tldr::TldrLib& tldr) {
    std::string input;
    std::map<std::string, std::function<void(const std::string &)> > actions = {
        {"do-rag", [&tldr](const std::string &arg) { tldr.doRag(arg); }},
        {"add-corpus", [&tldr](const std::string &arg) { tldr.processDocument(arg); }},
        {"delete-corpus", [&tldr](const std::string &arg) { tldr.removeFromDatabase(arg); }},
        {"query", [&tldr](const std::string &arg) { 
            tldr.queryRag(arg, EMBEDDINGS_URL, CHAT_URL); 
        }}
    };

    while (true) {
        std::cout << "Enter command: ";
        std::getline(std::cin, input);
        std::istringstream iss(input);
        std::string command, argument;
        iss >> command >> argument;

        auto it = actions.find(command);
        if (it != actions.end()) {
            try {
                it->second(argument);
            } catch (const std::exception& e) {
                std::cerr << "Error executing command: " << e.what() << std::endl;
            }
        } else if (command == "exit") {
            break;
        } else {
            std::cout << "Unknown command: " << command << std::endl;
            std::cout << "Available commands: do-rag, add-corpus, delete-corpus, query, exit" << std::endl;
        }
    }
}

int main() {
    try {
        // Initialize TldrLib
        tldr::TldrLib tldr;
        
        // Initialize database
        std::string dbPath = tldr::TldrLib::translatePath(DB_PATH);
        tldr.initializeDatabase(dbPath);

        // Test with a sample file
        std::string testFile = tldr::TldrLib::translatePath(
            "~/proj_tldr/corpus/current/97-things-every-software-architect-should-know.pdf");
        std::cout << "Testing addCorpus with file: " << testFile << std::endl;

        // Process the test file
        tldr.processDocument(testFile);

        // Test query
        tldr.queryRag("What does the book say about the practice of commit-and-run?", 
                     EMBEDDINGS_URL, 
                     CHAT_URL);

        // Start interactive command loop
        command_loop(tldr);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
