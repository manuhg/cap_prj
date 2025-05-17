import Foundation
import SwiftUI
import TldrAPI

class ChatViewModel: ObservableObject {
    @Published var conversations: [Conversation] = []
    @Published var selectedConversation: Conversation?
    @Published var newMessageText: String = ""
    @Published var isLoading: Bool = false
    @Published var errorMessage: String? = nil
    
    private let corpusDir = "/Users/manu/proj_tldr/corpus/current/"
    
    init() {
        // Initialize the TLDR system
        if !TldrWrapper.initialize() {
            errorMessage = "Failed to initialize TLDR system"
        }
        
        // Load sample conversations
        loadSampleData()
    }
    
    deinit {
        // Clean up the TLDR system
        TldrWrapper.cleanup()
    }
    
    private func loadSampleData() {
        // Start with an empty conversation
        let newConversation = Conversation(title: "New Conversation")
        conversations = [newConversation]
        selectedConversation = newConversation
    }
    
    func sendMessage() {
        guard !newMessageText.isEmpty, 
              let conversation = selectedConversation,
              let conversationIndex = conversations.firstIndex(where: { $0.id == conversation.id }) else { 
            return 
        }
        
        // Add user message
        let userMessage = Message(content: newMessageText, sender: .user)
        var updatedConversation = Conversation(
            id: conversation.id,
            title: conversation.title,
            messages: conversation.messages + [userMessage],
            lastUpdated: Date()
        )
        
        // Update UI immediately
        conversations[conversationIndex] = updatedConversation
        selectedConversation = updatedConversation
        
        // Clear input field
        let userQuery = newMessageText
        newMessageText = ""
        
        // Show loading indicator
        isLoading = true
        errorMessage = nil
        
        // Perform RAG query in background
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            // Query the RAG system
            if let result = TldrWrapper.queryRag(userQuery, corpusDir: self.corpusDir) {
                DispatchQueue.main.async {
                    self.handleRagResult(result, for: conversation, at: conversationIndex)
                }
            } else {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to get response from RAG system"
                    self.isLoading = false
                }
            }
        }
    }
    
    private func handleRagResult(_ result: RagResult, for conversation: Conversation, at index: Int) {
        // Create assistant message with the response
        var responseText = result.response
        
        // Optionally include context chunks in the response
        if !result.contextChunks.isEmpty {
            responseText += "\n\nContext used:\n"
            for (i, chunk) in result.contextChunks.prefix(3).enumerated() {
                responseText += "\n\(i + 1). [Similarity: \(String(format: "%.2f", chunk.similarity * 100))%]"
            }
        }
        
        let assistantMessage = Message(content: responseText, sender: .assistant)
        
        // Update conversation with assistant's response
        var updatedConversation = conversation
        updatedConversation.messages.append(assistantMessage)
        updatedConversation.lastUpdated = Date()
        
        // Update UI
        conversations[index] = updatedConversation
        selectedConversation = updatedConversation
        isLoading = false
    }
    
    func createNewConversation() {
        let newConversation = Conversation(title: "New Conversation")
        conversations.append(newConversation)
        selectedConversation = newConversation
    }
} 
