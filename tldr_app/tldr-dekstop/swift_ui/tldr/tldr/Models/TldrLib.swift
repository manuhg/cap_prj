import Foundation

class TldrLib {
    private var wrapper: OpaquePointer?
    
    init() {
        wrapper = tldr_create()
    }
    
    deinit {
        if let wrapper = wrapper {
            tldr_destroy(wrapper)
        }
    }
    
    func initializeDatabase(path: String) throws {
        guard let wrapper = wrapper else {
            throw TldrError.notInitialized
        }
        
        let result = tldr_initialize_database(wrapper, path)
        if result != 0 {
            throw TldrError.databaseError
        }
    }
    
    func processDocument(filePath: String) throws {
        guard let wrapper = wrapper else {
            throw TldrError.notInitialized
        }
        
        let result = tldr_process_document(wrapper, filePath)
        if result != 0 {
            throw TldrError.processingError
        }
    }
    
    func generateSummary(text: String) throws -> String {
        guard let wrapper = wrapper else {
            throw TldrError.notInitialized
        }
        
        guard let summary = tldr_generate_summary(wrapper, text) else {
            throw TldrError.summaryError
        }
        defer { tldr_free_string(summary) }
        
        return String(cString: summary)
    }
    
    func askQuestion(question: String, context: String) throws -> String {
        guard let wrapper = wrapper else {
            throw TldrError.notInitialized
        }
        
        guard let answer = tldr_ask_question(wrapper, question, context) else {
            throw TldrError.questionError
        }
        defer { tldr_free_string(answer) }
        
        return String(cString: answer)
    }
    
    func searchDocuments(query: String) throws -> [String] {
        guard let wrapper = wrapper else {
            throw TldrError.notInitialized
        }
        
        var count: Int32 = 0
        guard let results = tldr_search_documents(wrapper, query, &count) else {
            return []
        }
        defer { tldr_free_string_array(results, count) }
        
        var documents: [String] = []
        for i in 0..<Int(count) {
            if let str = results.advanced(by: i).pointee {
                documents.append(String(cString: str))
            }
        }
        
        return documents
    }
    
    func addToDatabase(text: String) throws {
        guard let wrapper = wrapper else {
            throw TldrError.notInitialized
        }
        
        let result = tldr_add_to_database(wrapper, text)
        if result != 0 {
            throw TldrError.databaseError
        }
    }
    
    func removeFromDatabase(filePath: String) throws {
        guard let wrapper = wrapper else {
            throw TldrError.notInitialized
        }
        
        let result = tldr_remove_from_database(wrapper, filePath)
        if result != 0 {
            throw TldrError.databaseError
        }
    }
}

enum TldrError: Error {
    case notInitialized
    case databaseError
    case processingError
    case summaryError
    case questionError
}
