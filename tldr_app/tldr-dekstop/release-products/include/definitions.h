#ifndef TLDR_CPP_DEFINITIONS_H
#define TLDR_CPP_DEFINITIONS_H
#include <vector>
#include <string>

// Structure for operation results
struct WorkResult {
    bool error{false};
    std::string error_message{};
    std::string success_message{};

    // Helper function to create an error result
    static WorkResult Error(const std::string &message) {
        return {true, message};
    }

    // Implicit bool conversion for easy error checking
    operator bool() const { return !error; }
};

// Structure for similarity search results from the NPU accelerator
struct VectorSimilarityMatch {
    uint64_t hash;
    float score;
    std::string text; // Text associated with the hash (filled in by wrapper)
};

struct CtxChunkMeta {
    std::string text;
    float similarity;
    uint64_t hash;

    // Document metadata
    std::string file_path;
    std::string file_name;
    std::string title;
    std::string author;
    int page_count;

    // Page number this chunk belongs to
    int page_number = 0;
};

// Wrapper function for NPU similarity search
std::vector<CtxChunkMeta> searchSimilarVectorsNPU(
    const std::vector<float> &query_vector,
    const std::string &corpus_dir,
    int k
);

// Structure to hold PDF metadata
struct PdfMetadata {
    std::string title;
    std::string author;
    std::string subject;
    std::string keywords;
    std::string creator;
    std::string producer;
    int pageCount;
};

// Structure to hold document data including metadata and page texts
struct DocumentData {
    PdfMetadata metadata;
    std::vector<std::string> pageTexts; // Index N-1 contains text of page N
    std::vector<std::string> chunks; // Text chunks for processing
    std::vector<int> chunkPageNums; // Page number for each chunk
};


// Structure to hold RAG query results
struct RagResult {
    // The generated response from the LLM
    std::string response;

    // The chunks used as context for the query
    std::vector<CtxChunkMeta> context_chunks;

    // Number of documents referenced in the result
    int referenced_document_count = 0;
};

struct embeddings_request {
    std::vector<std::string> input;
};

#endif// TLDR_CPP_DEFINITIONS_H
