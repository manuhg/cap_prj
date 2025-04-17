import SwiftUI

struct ChatView: View {
    let conversation: Conversation
    @ObservedObject var viewModel: ChatViewModel
    
    var body: some View {
        VStack {
            // Messages
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(conversation.messages) { message in
                        MessageBubble(message: message)
                    }
                }
                .padding()
            }
            
            // Input area
            HStack {
                TextField("Type a message...", text: $viewModel.newMessageText, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...5)
                
                Button(action: viewModel.sendMessage) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .disabled(viewModel.newMessageText.isEmpty)
            }
            .padding()
        }
        .navigationTitle(conversation.title)
    }
}

struct MessageBubble: View {
    let message: Message
    
    var body: some View {
        HStack {
            if message.sender == .assistant {
                Spacer()
            }
            
            VStack(alignment: message.sender == .user ? .trailing : .leading) {
                Text(message.content)
                    .padding()
                    .background(message.sender == .user ? Color.blue : Color.gray.opacity(0.2))
                    .foregroundColor(message.sender == .user ? .white : .primary)
                    .cornerRadius(12)
                
                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            if message.sender == .user {
                Spacer()
            }
        }
    }
} 