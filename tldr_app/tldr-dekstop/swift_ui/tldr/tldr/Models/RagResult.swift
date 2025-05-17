import Foundation

/// Represents a context chunk from a RAG query result
public struct ContextChunk: Identifiable {
    public let id: UInt64  // Using hash as unique identifier
    public let text: String
    public let similarity: Float
    
    init(_ chunk: TldrAPI.ContextChunk) {
        self.id = chunk.hash
        self.text = String(cString: chunk.text)
        self.similarity = chunk.similarity
    }
}

/// Represents the result of a RAG query
public struct RagResult {
    /// The generated response from the LLM
    public let response: String
    
    /// The context chunks used to generate the response
    public let contextChunks: [ContextChunk]
    
    /// Initialize from C RagResult
    init?(_ cResult: UnsafePointer<TldrAPI.RagResult>?) {
        guard let cResult = cResult?.pointee else { return nil }
        
        // Copy the response string
        self.response = String(cString: cResult.response)
        
        // Copy context chunks
        var chunks: [ContextChunk] = []
        for i in 0..<cResult.context_chunks_count {
            let cChunk = cResult.context_chunks[Int(i)]
            chunks.append(ContextChunk(cChunk))
        }
        self.contextChunks = chunks
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
    
    /// Add a single PDF file to the corpus
    public static func addFileToCorpus(_ path: String) {
        path.withCString { cString in
            TldrAPI.tldr_addFileToCorpus(cString)
        }
    }
    
    /// Delete a corpus by ID
    public static func deleteCorpus(_ id: String) {
        id.withCString { cString in
            TldrAPI.tldr_deleteCorpus(cString)
        }
    }
    
    /// Query the RAG system
    public static func queryRag(_ query: String, corpusDir: String? = nil) -> RagResult? {
        return query.withCString { queryCString in
            let cResult: UnsafeMutablePointer<TldrAPI.RagResult>?
            
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
            
            return RagResult(cResult)
        }
    }
    
    /// Test function to verify the bridge is working
    public static func testBridge() -> Int32 {
        return TldrAPI.tldr_api_trial_tldr()
    }
}
