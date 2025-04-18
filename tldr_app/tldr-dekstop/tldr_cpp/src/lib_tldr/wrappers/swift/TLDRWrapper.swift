import Foundation

class TLDRWrapper {
    static let shared = TLDRWrapper()
    
    private init() {}
    
    func initializeSystem() -> Bool {
        return tldr_initialize_system()
    }
    
    func cleanupSystem() {
        tldr_cleanup_system()
    }
    
    func addCorpus(sourcePath: String) {
        tldr_add_corpus(sourcePath)
    }
    
    func deleteCorpus(corpusId: String) {
        tldr_delete_corpus(corpusId)
    }
    
    func queryRag(userQuery: String, 
                 embeddingsUrl: String = "http://localhost:8084/embeddings",
                 chatUrl: String = "http://localhost:8088/v1/chat/completions") {
        tldr_query_rag(userQuery, embeddingsUrl, chatUrl)
    }
    
    func translatePath(path: String) -> String {
        let cstr = tldr_translate_path(path)
        let result = String(cString: cstr)
        // Free the C string
        cstr.deallocate()
        return result
    }
} 