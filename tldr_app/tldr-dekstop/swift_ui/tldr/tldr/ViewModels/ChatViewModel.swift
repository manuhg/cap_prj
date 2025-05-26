import Foundation
import SwiftUI
import AppKit
import TldrAPI

@MainActor
class ChatViewModel: ObservableObject {
    @Published var conversations: [ConversationData] = []
    @Published var selectedConversation: ConversationData?
    @Published var newMessageText: String = ""
    @Published var isLoading: Bool = false
    @Published var errorMessage: String? = nil

    
    private let conversationsKey = "savedConversations"
    private let selectedConversationIdKey = "selectedConversationId"
    private let defaultCorpusDir = "~/Downloads"
    
    init() {
        print("[DEBUG] Initializing ChatViewModel")
        // Initialize the TLDR system
        let initResult = TldrWrapper.initialize()
        print("[DEBUG] TldrWrapper.initialize() result: \(initResult)")
        
        if !initResult {
            print("[DEBUG] Failed to initialize TLDR system")
            errorMessage = "Failed to initialize TLDR system"
        } else {
            print("[DEBUG] TLDR system initialized successfully")
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
            print("[DEBUG] Loaded conversations: \(savedConversations.map { "\($0.title): \($0.corpusDir)" })")
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
            print("[DEBUG] Saved conversations: \(conversations.map { "\($0.title): \($0.corpusDir)" })")
        }
        
        // Save selected conversation ID
        if let selectedId = selectedConversation?.id {
            UserDefaults.standard.set(selectedId.uuidString, forKey: selectedConversationIdKey)
            print("[DEBUG] Saved selected conversation ID: \(selectedId)")
        }
    }
    
    func updateCorpusDirectory(_ newPath: String) {
        guard let conversation = selectedConversation,
              let index = conversations.firstIndex(where: { $0.id == conversation.id }) else {
            return
        }
        
        var updatedConversation = conversation
        updatedConversation.corpusDir = newPath
        updatedConversation.lastUpdated = Date()
        
        // Add a system message about the change
        updatedConversation.messages.append(
            Message(content: "Corpus directory changed to: " + newPath, sender: .system)
        )
        
        conversations[index] = updatedConversation
        selectedConversation = updatedConversation
        saveConversations()
    }
    
    func sendMessage() {
        print("[DEBUG] sendMessage called with text: \(newMessageText)")
        guard !newMessageText.isEmpty, 
              let conversation = selectedConversation,
              let conversationIndex = conversations.firstIndex(where: { $0.id == conversation.id }) else { 
            print("[DEBUG] sendMessage guard check failed")
            return 
        }
        
        print("[DEBUG] Using corpus directory: \(conversation.corpusDir)")
        
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
            print("[DEBUG] Adding user message to conversation")
            
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
            print("[DEBUG] Starting RAG query in background thread")
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self = self else { return }
                
                print("[DEBUG] Querying RAG system with message: \(messageText)")
                print("[DEBUG] Using corpus directory: \(conversation.corpusDir)")
                
                // Query the RAG system using the conversation's corpusDir
                if let result = TldrWrapper.queryRag(messageText, corpusDir: conversation.corpusDir) {
                    print("[DEBUG] RAG query successful, response length: \(result.response.count)")
                    DispatchQueue.main.async { [weak self] in
                        guard let self = self else { return }
                        print("[DEBUG] Handling RAG result on main thread")
                        self.handleRagResult(result, for: conversationIndex)
                    }
                } else {
                    print("[DEBUG] RAG query failed, result is nil")
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
        print("[DEBUG] handleRagResult called with result")
        
        // Get formatted response
        let responseText = result.formattedString() ?? result.response
        print("[DEBUG] Response text length: \(responseText.count)")
        print("[DEBUG] Response text preview: \(responseText.prefix(100))")
        
        let assistantMessage = Message(content: responseText, sender: .assistant)
        
        // Update conversation with assistant's response
        var updatedConversation = conversations[conversationIndex]
        print("[DEBUG] Adding assistant message to conversation \(updatedConversation.id)")
        updatedConversation.messages.append(assistantMessage)
        updatedConversation.lastUpdated = Date()
        
        // Update UI
        print("[DEBUG] Updating UI with new conversation data")
        conversations[conversationIndex] = updatedConversation
        selectedConversation = updatedConversation
        
        // Save changes
        print("[DEBUG] Saving conversations to UserDefaults")
        saveConversations()
        
        print("[DEBUG] Setting isLoading to false")
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
    

    
    @MainActor func selectCorpusDirectory() {
        Task {
            let openPanel = NSOpenPanel()
            openPanel.canChooseFiles = false
            openPanel.canChooseDirectories = true
            openPanel.allowsMultipleSelection = false
            openPanel.message = "Select a corpus directory"
            
            if openPanel.runModal() == .OK {
                if let url = openPanel.url {
                    await MainActor.run {
                        updateCorpusDirectory(url.path)
                    }
                }
            }
        }
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
