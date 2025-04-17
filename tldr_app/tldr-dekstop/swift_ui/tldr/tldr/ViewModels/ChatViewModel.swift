import Foundation
import SwiftUI

class ChatViewModel: ObservableObject {
    @Published var conversations: [Conversation] = []
    @Published var selectedConversation: Conversation?
    @Published var newMessageText: String = ""
    @Published var isProcessing: Bool = false
    @Published var error: String? = nil
    
    private let tldrLib: TldrLib
    
    init() {
        tldrLib = TldrLib()
        do {
            try tldrLib.initializeDatabase(path: "tldr.db")
        } catch {
            self.error = "Failed to initialize database: \(error.localizedDescription)"
        }
    }
    
    func sendMessage() {
        guard !newMessageText.isEmpty, let conversation = selectedConversation else { return }
        
        let userMessage = Message(content: newMessageText, sender: .user)
        let userMessageText = newMessageText
        newMessageText = ""
        
        var updatedMessages = conversation.messages + [userMessage]
        let updatedConversation = Conversation(
            id: conversation.id,
            title: conversation.title,
            messages: updatedMessages,
            lastUpdated: Date()
        )
        
        updateConversation(updatedConversation)
        
        isProcessing = true
        error = nil
        
        Task {
            do {
                let context = updatedMessages
                    .map { "\($0.sender == .user ? "User" : "Assistant"): \($0.content)" }
                    .joined(separator: "\n")
                
                let answer = try await Task.detached { [self] in
                    try self.tldrLib.askQuestion(question: userMessageText, context: context)
                }.value
                
                await MainActor.run {
                    let assistantMessage = Message(content: answer, sender: .assistant)
                    updatedMessages.append(assistantMessage)
                    
                    let finalConversation = Conversation(
                        id: conversation.id,
                        title: conversation.title,
                        messages: updatedMessages,
                        lastUpdated: Date()
                    )
                    
                    updateConversation(finalConversation)
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    self.error = error.localizedDescription
                    isProcessing = false
                }
            }
        }
    }
    
    private func updateConversation(_ conversation: Conversation) {
        if let index = conversations.firstIndex(where: { $0.id == conversation.id }) {
            conversations[index] = conversation
            selectedConversation = conversation
        }
    }
    
    func createNewConversation() {
        let newConversation = Conversation(title: "New Conversation")
        conversations.append(newConversation)
        selectedConversation = newConversation
    }
    
    func processDocument(at url: URL) {
        isProcessing = true
        error = nil
        
        Task {
            do {
                try await Task.detached { [self] in
                    try self.tldrLib.processDocument(filePath: url.path)
                }.value
                
                await MainActor.run {
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    self.error = "Failed to process document: \(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }
}