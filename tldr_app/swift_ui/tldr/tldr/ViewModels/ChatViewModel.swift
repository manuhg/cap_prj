import Foundation
import SwiftUI

class ChatViewModel: ObservableObject {
    @Published var conversations: [Conversation] = []
    @Published var selectedConversation: Conversation?
    @Published var newMessageText: String = ""
    
    init() {
        // Load sample data for testing
        loadSampleData()
    }
    
    private func loadSampleData() {
        conversations = [
            Conversation(title: "First Conversation"),
            Conversation(title: "Second Conversation")
        ]
        selectedConversation = conversations.first
    }
    
    func sendMessage() {
        guard !newMessageText.isEmpty, let conversation = selectedConversation else { return }
        
        let newMessage = Message(content: newMessageText, sender: .user)
        let updatedConversation = Conversation(
            id: conversation.id,
            title: conversation.title,
            messages: conversation.messages + [newMessage],
            lastUpdated: Date()
        )
        
        if let index = conversations.firstIndex(where: { $0.id == conversation.id }) {
            conversations[index] = updatedConversation
            selectedConversation = updatedConversation
        }
        
        newMessageText = ""
    }
    
    func createNewConversation() {
        let newConversation = Conversation(title: "New Conversation")
        conversations.append(newConversation)
        selectedConversation = newConversation
    }
} 