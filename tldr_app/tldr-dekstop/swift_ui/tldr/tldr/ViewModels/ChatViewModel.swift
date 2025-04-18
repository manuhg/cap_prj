import Foundation
import SwiftUI
import TldrAPI


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
        
        let userMessage = Message(content: newMessageText, sender: .user)
        var updatedConversation = Conversation(
            id: conversation.id,
            title: conversation.title,
            messages: conversation.messages + [userMessage],
            lastUpdated: Date()
        )
        
        // Call TLDR backend via native Swift wrapper
        TldrAPI.queryRag(userQuery: newMessageText) // This call may need to be async if the wrapper is updated for async/await
        let assistantMessage = Message(content: "[TLDR response here]", sender: .assistant) // Replace with actual response if queryRag returns a value
        updatedConversation.messages.append(assistantMessage)
        updatedConversation.lastUpdated = Date()
        
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
