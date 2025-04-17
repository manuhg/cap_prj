#include "../tldr_wrapper.h"
#include "tldr_lib.h"
#include <cstring>
#include <string>
#include <vector>

struct TldrLibWrapper {
    tldr::TldrLib* lib;
    
    TldrLibWrapper() : lib(new tldr::TldrLib()) {}
    ~TldrLibWrapper() { delete lib; }
};

TldrLibWrapper* tldr_create() {
    return new TldrLibWrapper();
}

void tldr_destroy(TldrLibWrapper* wrapper) {
    delete wrapper;
}

int tldr_initialize_database(TldrLibWrapper* wrapper, const char* db_path) {
    try {
        wrapper->lib->initializeDatabase(db_path);
        return 0;
    } catch (...) {
        return -1;
    }
}

int tldr_process_document(TldrLibWrapper* wrapper, const char* file_path) {
    try {
        wrapper->lib->processDocument(file_path);
        return 0;
    } catch (...) {
        return -1;
    }
}

char* tldr_generate_summary(TldrLibWrapper* wrapper, const char* text) {
    try {
        auto summaries = wrapper->lib->generateSummary(text);
        std::string combined;
        for (const auto& summary : summaries) {
            if (!combined.empty()) combined += "\n";
            combined += summary;
        }
        char* result = strdup(combined.c_str());
        return result;
    } catch (...) {
        return nullptr;
    }
}

char* tldr_ask_question(TldrLibWrapper* wrapper, const char* question, const char* context) {
    try {
        std::string answer = wrapper->lib->askQuestion(question, context);
        return strdup(answer.c_str());
    } catch (...) {
        return nullptr;
    }
}

char** tldr_search_documents(TldrLibWrapper* wrapper, const char* query, int* count) {
    try {
        auto results = wrapper->lib->searchDocuments(query);
        *count = static_cast<int>(results.size());
        if (*count == 0) return nullptr;
        
        char** array = new char*[*count];
        for (int i = 0; i < *count; i++) {
            array[i] = strdup(results[i].c_str());
        }
        return array;
    } catch (...) {
        *count = 0;
        return nullptr;
    }
}

int tldr_add_to_database(TldrLibWrapper* wrapper, const char* text) {
    try {
        wrapper->lib->addToDatabase(text);
        return 0;
    } catch (...) {
        return -1;
    }
}

int tldr_remove_from_database(TldrLibWrapper* wrapper, const char* file_path) {
    try {
        wrapper->lib->removeFromDatabase(file_path);
        return 0;
    } catch (...) {
        return -1;
    }
}

void tldr_free_string(char* str) {
    free(str);
}

void tldr_free_string_array(char** array, int count) {
    if (array) {
        for (int i = 0; i < count; i++) {
            free(array[i]);
        }
        delete[] array;
    }
}
