import Foundation
import SwiftUI
import TldrAPI

class ChatViewModel: ObservableObject {
    @Published var conversations: [Conversation] = []
    @Published var selectedConversation: Conversation?
    @Published var newMessageText: String = ""
    @Published var isLoading: Bool = false
    @Published var errorMessage: String? = nil
    @Published var corpusDir: String
    @Published var showingCorpusDialog: Bool = false
    
    private let conversationsKey = "savedConversations"
    private let selectedConversationIdKey = "selectedConversationId"
    private let corpusDirKey = "corpusDirectory"
    
    init() {
        // Load saved corpus directory or use default
        self.corpusDir = UserDefaults.standard.string(forKey: corpusDirKey) ?? "/Users/manu/proj_tldr/corpus/current/"
        
        // Initialize the TLDR system
        if !TldrWrapper.initialize() {
            errorMessage = "Failed to initialize TLDR system"
        }
        
        // Load saved conversations
        loadSavedConversations()
    }
    
    deinit {
        // Clean up the TLDR system
        TldrWrapper.cleanup()
    }
    
    private func loadSavedConversations() {
        // Load conversations from UserDefaults
        if let data = UserDefaults.standard.data(forKey: conversationsKey),
           let savedConversations = try? JSONDecoder().decode([Conversation].self, from: data) {
            conversations = savedConversations
        } else {
            // If no saved conversations, create a new one
            let newConversation = Conversation(title: "New Conversation")
            conversations = [newConversation]
        }
        
        // Load selected conversation ID
        if let selectedId = UserDefaults.standard.string(forKey: selectedConversationIdKey),
           let uuid = UUID(uuidString: selectedId) {
            selectedConversation = conversations.first { $0.id == uuid }
        }
        
        // If no selected conversation, select the first one
        if selectedConversation == nil {
            selectedConversation = conversations.first
        }
    }
    
    private func saveConversations() {
        // Save conversations to UserDefaults
        if let data = try? JSONEncoder().encode(conversations) {
            UserDefaults.standard.set(data, forKey: conversationsKey)
        }
        
        // Save selected conversation ID
        if let selectedId = selectedConversation?.id {
            UserDefaults.standard.set(selectedId.uuidString, forKey: selectedConversationIdKey)
        }
    }
    
    func updateCorpusDirectory(_ newPath: String) {
        corpusDir = newPath
        UserDefaults.standard.set(newPath, forKey: corpusDirKey)
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
        
        // Save changes
        saveConversations()
        
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
    
    private func handleRagResult(_ result: RagResultSw, for conversation: Conversation, at index: Int) {
        // Get formatted response
        let responseText = result.formattedString() ?? result.response
        
        let assistantMessage = Message(content: responseText, sender: .assistant)
        
        // Update conversation with assistant's response
        var updatedConversation = conversation
        updatedConversation.messages.append(assistantMessage)
        updatedConversation.lastUpdated = Date()
        
        // Update UI
        conversations[index] = updatedConversation
        selectedConversation = updatedConversation
        
        // Save changes
        saveConversations()
        
        isLoading = false
    }
    
    func createNewConversation() {
        let newConversation = Conversation(title: "New Conversation")
        conversations.append(newConversation)
        selectedConversation = newConversation
        
        // Save changes
        saveConversations()
    }
    
    func deleteConversation(_ conversation: Conversation) {
        conversations.removeAll { $0.id == conversation.id }
        
        // If we deleted the selected conversation, select another one
        if selectedConversation?.id == conversation.id {
            selectedConversation = conversations.first
        }
        
        // Save changes
        saveConversations()
    }
    
    func updateConversationTitle(_ conversation: Conversation, newTitle: String) {
        if let index = conversations.firstIndex(where: { $0.id == conversation.id }) {
            var updatedConversation = conversation
            updatedConversation.title = newTitle
            conversations[index] = updatedConversation
            
            // If this is the selected conversation, update it
            if selectedConversation?.id == conversation.id {
                selectedConversation = updatedConversation
            }
            
            // Save changes
            saveConversations()
        }
    }
} 
