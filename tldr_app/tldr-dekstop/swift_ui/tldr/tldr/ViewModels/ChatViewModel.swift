import Foundation
import SwiftUI
import TldrAPI

@MainActor
class ChatViewModel: ObservableObject {
    @Published var conversations: [ConversationData] = []
    @Published var selectedConversation: ConversationData?
    @Published var newMessageText: String = ""
    @Published var isLoading: Bool = false
    @Published var errorMessage: String? = nil
    @Published var showingCorpusDialog: Bool = false
    
    private let conversationsKey = "savedConversations"
    private let selectedConversationIdKey = "selectedConversationId"
    private let defaultCorpusDir = "~/Downloads"
    
    init() {
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
           let savedConversations = try? JSONDecoder().decode([ConversationData].self, from: data) {
            conversations = savedConversations
            
            // Load selected conversation ID
            if let selectedId = UserDefaults.standard.string(forKey: selectedConversationIdKey),
               let uuid = UUID(uuidString: selectedId) {
                selectedConversation = conversations.first { $0.id == uuid }
            }
            
            // If no selected conversation, select the first one
            if selectedConversation == nil {
                selectedConversation = conversations.first
            }
        } else {
            // If no saved conversations, create a new one with default corpus dir and welcome message
            let newConversation = ConversationData(
                title: "New Conversation",
                corpusDir: defaultCorpusDir,
                messages: [
                    Message(content: "Welcome to TLDR! I'll help you understand your codebase. Start by selecting a corpus directory using the folder icon above.", sender: .system)
                ]
            )
            conversations = [newConversation]
            selectedConversation = newConversation
            saveConversations()
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
        guard let conversation = selectedConversation,
              let index = conversations.firstIndex(where: { $0.id == conversation.id }) else {
            return
        }
        
        var updatedConversation = conversation
        updatedConversation.corpusDir = newPath
        conversations[index] = updatedConversation
        selectedConversation = updatedConversation
        saveConversations()
    }
    
    func sendMessage() {
        guard !newMessageText.isEmpty, 
              let conversation = selectedConversation,
              let conversationIndex = conversations.firstIndex(where: { $0.id == conversation.id }) else { 
            return 
        }
        
        // Create a local copy of the message text
        let messageText = newMessageText
        
        // Clear input field immediately
        newMessageText = ""
        
        // Show loading indicator
        isLoading = true
        errorMessage = nil
        
        // Schedule the rest of the updates for the next run loop
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            // Add user message
            let userMessage = Message(content: messageText, sender: .user)
            var updatedConversation = conversation
            updatedConversation.messages.append(userMessage)
            updatedConversation.lastUpdated = Date()
            
            // Update UI
            self.conversations[conversationIndex] = updatedConversation
            self.selectedConversation = updatedConversation
            self.saveConversations()
            
            // Perform RAG query in background
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self = self else { return }
                
                // Query the RAG system using the conversation's corpusDir
                if let result = TldrWrapper.queryRag(messageText, corpusDir: conversation.corpusDir) {
                    DispatchQueue.main.async { [weak self] in
                        guard let self = self else { return }
                        self.handleRagResult(result, for: conversationIndex)
                    }
                } else {
                    DispatchQueue.main.async { [weak self] in
                        guard let self = self else { return }
                        self.errorMessage = "Failed to get response from RAG system"
                        self.isLoading = false
                    }
                }
            }
        }
    }
    
    private func handleRagResult(_ result: RagResultSw, for conversationIndex: Int) {
        // Get formatted response
        let responseText = result.formattedString() ?? result.response
        let assistantMessage = Message(content: responseText, sender: .assistant)
        
        // Update conversation with assistant's response
        var updatedConversation = conversations[conversationIndex]
        updatedConversation.messages.append(assistantMessage)
        updatedConversation.lastUpdated = Date()
        
        // Update UI
        conversations[conversationIndex] = updatedConversation
        selectedConversation = updatedConversation
        
        // Save changes
        saveConversations()
        
        isLoading = false
    }
    
    func deleteConversation(_ conversation: ConversationData) {
        // Remove the conversation from the list
        conversations.removeAll { $0.id == conversation.id }
        
        // If this was the selected conversation, select another one
        if selectedConversation?.id == conversation.id {
            selectedConversation = conversations.first
        }
        
        // Save changes
        saveConversations()
    }
    
    func createNewConversation() {
        // Use the corpus directory from the currently selected conversation, or the default if none is selected
        let corpusDir = selectedConversation?.corpusDir ?? defaultCorpusDir
        let newConversation = ConversationData(
            title: "New Conversation",
            corpusDir: corpusDir,
            messages: [
                Message(content: "New conversation started. Using corpus directory: " + corpusDir, sender: .system)
            ]
        )
        conversations.append(newConversation)
        selectedConversation = newConversation
        
        // Save changes
        saveConversations()
    }
    

    
    func updateConversationCorpusDirectory(_ newPath: String) {
        guard let conversation = selectedConversation,
              let index = conversations.firstIndex(where: { $0.id == conversation.id }) else {
            return
        }
        
        var updatedConversation = conversation
        updatedConversation.corpusDir = newPath
        
        // Add a system message about the change
        updatedConversation.messages.append(
            Message(content: "Corpus directory changed to: " + newPath, sender: .system)
        )
        
        conversations[index] = updatedConversation
        selectedConversation = updatedConversation
        
        // Save changes
        saveConversations()
    }
    
    func updateConversationTitle(_ conversation: ConversationData, newTitle: String) {
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
