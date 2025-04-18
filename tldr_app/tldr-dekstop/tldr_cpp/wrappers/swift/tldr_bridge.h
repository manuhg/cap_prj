#ifndef TLDR_BRIDGE_H
#define TLDR_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the TLDR system
bool tldr_initialize_system(void);

// Clean up the TLDR system
void tldr_cleanup_system(void);

// Add a document to the corpus
void tldr_add_corpus(const char* source_path);

// Delete a document from the corpus
void tldr_delete_corpus(const char* corpus_id);

// Query the RAG system
void tldr_query_rag(const char* user_query, 
                    const char* embeddings_url,
                    const char* chat_url);

// Translate a path with environment variables and home directory expansion
char* tldr_translate_path(const char* path);

#ifdef __cplusplus
}
#endif

#endif // TLDR_BRIDGE_H 