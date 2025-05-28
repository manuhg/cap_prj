import Foundation

struct Message: Identifiable, Codable {
    let id: UUID
    let content: String
    let timestamp: Date
    let sender: Sender
    let contextChunks: [ContextChunkSw]?
    
    enum Sender: String, Codable {
        case user
        case assistant
        case system
    }
    
    init(id: UUID = UUID(), content: String, timestamp: Date = Date(), sender: Sender, contextChunks: [ContextChunkSw]? = nil) {
        self.id = id
        self.content = content
        self.timestamp = timestamp
        self.sender = sender
        self.contextChunks = contextChunks
    }
    
    // Custom decoder init for backward compatibility
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        content = try container.decode(String.self, forKey: .content)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        sender = try container.decode(Sender.self, forKey: .sender)
        
        // Try to decode contextChunks, but if the key doesn't exist, set to nil
        contextChunks = try container.decodeIfPresent([ContextChunkSw].self, forKey: .contextChunks)
    }
    
    var hasSourceContext: Bool {
        return contextChunks != nil && !contextChunks!.isEmpty
    }
} 