#include "tldr_api.h"
#include "lib_tldr.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace tldr_cpp_api {

bool initializeSystem(const std::string& chat_model_path, const std::string& embeddings_model_path) {
    return ::initializeSystem(chat_model_path, embeddings_model_path);
}

void cleanupSystem() {
    ::cleanupSystem();
}
 
void addCorpus(const std::string& sourcePath) {
    ::addCorpus(sourcePath);
}

void deleteCorpus(const std::string& corpusId) {
    ::deleteCorpus(corpusId);
}

RagResult queryRag(const std::string& user_query, const std::string& corpus_dir) {
    // Call the global queryRag function
    return ::queryRag(user_query, corpus_dir);
}

std::string printRagResult(const RagResult& result) {
    std::stringstream formatted_result;
    
    // Add the LLM response
    formatted_result << "=== LLM Response ===\n\n" << result.response << "\n\n";
    
    // Add information about the number of contexts used
    formatted_result << "=== Context Information ===\n";
    formatted_result << "Referenced " << result.referenced_document_count << " document(s) with "
                     << result.context_chunks.size() << " context chunk(s)\n\n";
    
    // Add details for each context chunk
    formatted_result << "=== Context Details ===\n\n";
    
    int chunk_number = 1;
    for (const auto& chunk : result.context_chunks) {
        // Create a header for each chunk
        formatted_result << "--- Chunk " << chunk_number++ << " ---\n";
        
        // Add source information
        formatted_result << "Source: ";
        if (!chunk.title.empty()) {
            formatted_result << "Title: \"" << chunk.title << "\"";
        } else if (!chunk.file_name.empty()) {
            formatted_result << chunk.file_name;
        } else {
            formatted_result << "[Unknown Source]";
        }
        
        if (!chunk.author.empty()) {
            formatted_result << " Author: " << chunk.author;
        }
        
        if (chunk.page_count > 0) {
            formatted_result << " (" << chunk.page_count << " Pages)";
        }
        
        if (chunk.page_number > 0) {
            formatted_result << ", Page " << chunk.page_number;
        }
        formatted_result << "\n";
        
        // Add similarity score
        formatted_result << "Similarity: " << std::fixed << std::setprecision(4) << chunk.similarity << "\n";
        
        // Add file information if available
        if (!chunk.file_path.empty()) {
            formatted_result << "File path: " << chunk.file_path << "\n";
        }

        // Add the actual text content
        formatted_result << "Content:\n" << chunk.text << "\n\n";
    }
    
    return formatted_result.str();
}

} // namespace tldr_cpp_api


/*
// C linkage function for Swift to call
extern "C" {
    int tldr_api_trial_tldr() {
        std::cout << "TldrAPI trial function called from Swift" << std::endl;
        tldr_cpp_api::initializeSystem();
        return 42;
    }
}*/