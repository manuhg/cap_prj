import Foundation

struct Message: Identifiable, Codable {
    let id: UUID
    let content: String
    let timestamp: Date
    let sender: Sender
    
    enum Sender: String, Codable {
        case user
        case assistant
    }
    
    init(id: UUID = UUID(), content: String, timestamp: Date = Date(), sender: Sender) {
        self.id = id
        self.content = content
        self.timestamp = timestamp
        self.sender = sender
    }
} 