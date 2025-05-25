import Foundation

struct ConversationData: Identifiable, Codable, Hashable {
    let id: UUID
    var title: String
    var corpusDir: String
    var messages: [Message]
    var lastUpdated: Date
    
    init(id: UUID = UUID(), 
         title: String, 
         corpusDir: String = "~/Downloads", 
         messages: [Message] = [], 
         lastUpdated: Date = Date()) {
        self.id = id
        self.title = title
        self.corpusDir = corpusDir
        self.messages = messages
        self.lastUpdated = lastUpdated
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    static func == (lhs: ConversationData, rhs: ConversationData) -> Bool {
        lhs.id == rhs.id
    }
}
