#include <iostream>
#include "lib_tldr/tldr_api.h"

int main() {
    // Initialize system
    if (!tldr_cpp_api::initializeSystem(
        "/Users/manu/llm-weights/Llama-3.2-1B-Instruct-Q3_K_L-lms.gguf",
        "/Users/manu/llm-weights/embedding/all-MiniLM-L6-v2-Q8_0.gguf")){
        std::cerr << "Failed to initialize system" << std::endl;
        return 1;
    }
    // Add a single file
    std::string testFile = "~/proj_tldr/corpus/current/0.System Design Interview An Insider’s Guide by Alex Xu.pdf";
    tldr_cpp_api::addCorpus(testFile);

    // Add a folder
    tldr_cpp_api::addCorpus("~/proj_tldr/corpus/current");

    // Do RAG
    std::string query = "What is the hotspot problem in cache?";
    std::string corpus_dir = "~/proj_tldr/corpus/current";
    RagResult result = tldr_cpp_api::queryRag(query, corpus_dir, "/Users/manu/dev/UW/cap_prj/tldr_app/tldr-dekstop/release-products/artefacts/CosineSimilarityBatched.mlmodelc");
    
    // Format and print the result with all context metadata
    std::string formatted_result = tldr_cpp_api::printRagResult(result);
    std::cout << "\n\nRESULT\n\n" << formatted_result << std::endl;

    // Cleanup system
    tldr_cpp_api::cleanupSystem();

    return 0;
}
