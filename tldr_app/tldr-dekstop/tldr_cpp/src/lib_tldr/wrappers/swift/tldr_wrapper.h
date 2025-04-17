#ifndef TLDR_WRAPPER_H
#define TLDR_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to TldrLib
typedef struct TldrLibWrapper TldrLibWrapper;

// Create and destroy TldrLib instance
TldrLibWrapper* tldr_create(void);
void tldr_destroy(TldrLibWrapper* wrapper);

// Core functions
int tldr_initialize_database(TldrLibWrapper* wrapper, const char* db_path);
int tldr_process_document(TldrLibWrapper* wrapper, const char* file_path);
char* tldr_generate_summary(TldrLibWrapper* wrapper, const char* text);
char* tldr_ask_question(TldrLibWrapper* wrapper, const char* question, const char* context);
char** tldr_search_documents(TldrLibWrapper* wrapper, const char* query, int* count);
int tldr_add_to_database(TldrLibWrapper* wrapper, const char* text);
int tldr_remove_from_database(TldrLibWrapper* wrapper, const char* file_path);
void tldr_free_string(char* str);
void tldr_free_string_array(char** array, int count);

#ifdef __cplusplus
}
#endif

#endif // TLDR_WRAPPER_H
