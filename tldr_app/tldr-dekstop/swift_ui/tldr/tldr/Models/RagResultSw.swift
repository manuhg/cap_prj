import Foundation
import TldrAPI
/// Represents a context chunk from a RAG query result
public struct ContextChunkSw: Identifiable, Codable {
    public let id: UInt64  // Using hash as unique identifier
    public let text: String
    public let filePath: String
    public let fileName: String
    public let title: String
    public let author: String
    public let pageCount: Int32
    public let pageNumber: Int32
    public let similarity: Float
    
    init(_ chunk: TldrAPI.CtxChunkMetaC) {
        self.id = chunk.hash
        self.text = String(cString: chunk.text)
        self.filePath = String(cString: chunk.file_path)
        self.fileName = String(cString: chunk.file_name)
        self.title = String(cString: chunk.title)
        self.author = String(cString: chunk.author)
        self.pageCount = chunk.page_count
        self.pageNumber = chunk.page_number
        self.similarity = chunk.similarity
    }
    
    // Additional initializer for creating from stored data
    public init(id: UInt64, text: String, filePath: String, fileName: String, title: String, author: String, pageCount: Int32, pageNumber: Int32, similarity: Float) {
        self.id = id
        self.text = text
        self.filePath = filePath
        self.fileName = fileName
        self.title = title
        self.author = author
        self.pageCount = pageCount
        self.pageNumber = pageNumber
        self.similarity = similarity
    }
}

/// Represents the result of a RAG query
public struct RagResultSw: Codable {
    /// The generated response from the LLM
    public let response: String
    
    /// The context chunks used to generate the response
    public let contextChunks: [ContextChunkSw]
    
    /// The number of referenced documents
    public let referencedDocumentCount: Int32
    
    /// Initialize from C RagResult
    init?(_ cResult: UnsafePointer<TldrAPI.RagResultC>?) {
        guard let cResult = cResult?.pointee else { return nil }
        
        // Copy the response string
        self.response = String(cString: cResult.response)
        self.referencedDocumentCount = cResult.referenced_document_count
        
        // Copy context chunks
        var chunks: [ContextChunkSw] = []
        for i in 0..<cResult.context_chunks_count {
            let cChunk = cResult.context_chunks[Int(i)]
            if (cChunk.similarity>=0.4){
                chunks.append(ContextChunkSw(cChunk))
            }
        }
        self.contextChunks = chunks
    }
    
    /// Get a formatted string representation of the result with clickable PDF links
    public func formattedString() -> String {
        // Start with the original response
        var formattedResponse = response
        
        // Append sources section if there are context chunks
        if !contextChunks.isEmpty {
            formattedResponse += "\n\nSources:\n"
            
            // Use a set to track unique combinations of file path and page number
            // We'll use a tuple of (filePath, pageNumber) as the key
            var uniqueSources = Set<String>()
            var uniqueChunks: [ContextChunkSw] = []
            
            // Filter out duplicate sources
            for chunk in contextChunks {
                // Only process PDF files
                guard chunk.filePath.lowercased().hasSuffix(".pdf") else { continue }
                
                // Create a unique key for this source (filePath + pageNumber)
                let sourceKey = "\(chunk.filePath)_\(chunk.pageNumber)"
                
                // Only add this source if we haven't seen it before
                if !uniqueSources.contains(sourceKey) {
                    uniqueSources.insert(sourceKey)
                    uniqueChunks.append(chunk)
                }
            }
            
            // Number each unique chunk sequentially
            var sourceNumber = 1
            
            // Add each unique chunk as a numbered source
            for chunk in uniqueChunks {
                // Format the file path for display (just the filename)
                let displayName = chunk.fileName.isEmpty ? 
                    (chunk.filePath as NSString).lastPathComponent : 
                    chunk.fileName
                
                // Create the source text
                let pageDisplay = chunk.pageNumber > 0 ? "page \(chunk.pageNumber)" : ""
                let sourceText = displayName + (pageDisplay.isEmpty ? "" : " (\(pageDisplay))")
                
                // Escape the file path by encoding it for a URL
                let escapedPath = chunk.filePath.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? chunk.filePath
                
                let urlPath = "file://\(escapedPath)"
                let pageParam = chunk.pageNumber > 0 ? "#page=\(chunk.pageNumber)" : ""
                let fullUrl = urlPath + pageParam
                
                // Add the formatted source to the response
                formattedResponse += "\(sourceNumber). \(sourceText)\n   \(fullUrl)\n"
                
                // Increment the source number
                sourceNumber += 1
            }
        }
        
        return formattedResponse
    }
}

/// Provides a Swift-friendly interface to the TLDR C API
public class TldrWrapper {
    /// Initialize the TLDR system
    public static func initialize() -> Bool {
        return TldrAPI.tldr_initializeSystem()
    }
    
    /// Clean up the TLDR system
    public static func cleanup() {
        TldrAPI.tldr_cleanupSystem()
    }
    
    /// Add a corpus from a PDF file or directory
    public static func addCorpus(_ path: String) {
        path.withCString { cString in
            TldrAPI.tldr_addCorpus(cString)
        }
    }
    
    /// Delete a corpus by ID
    public static func deleteCorpus(_ id: String) {
        id.withCString { cString in
            TldrAPI.tldr_deleteCorpus(cString)
        }
    }
    
    /// Query the RAG system
    public static func queryRag(_ query: String, corpusDir: String? = nil) -> RagResultSw? {
        return query.withCString { queryCString in
            let cResult: UnsafeMutablePointer<TldrAPI.RagResultC>?
            
            if let corpusDir = corpusDir {
                    cResult = corpusDir.withCString { dirCString in
                    TldrAPI.tldr_queryRag(queryCString, dirCString)
                }
            } else {
                cResult = TldrAPI.tldr_queryRag(queryCString, nil)
            }
            
            defer {
                if let cResult = cResult {
                    TldrAPI.tldr_freeRagResult(cResult)
                }
            }
            
            return RagResultSw(cResult)
        }
    }
}
