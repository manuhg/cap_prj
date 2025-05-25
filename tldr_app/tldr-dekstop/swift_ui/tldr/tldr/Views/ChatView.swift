import SwiftUI

struct ChatView: View {
    let conversation: ConversationData
    @ObservedObject var viewModel: ChatViewModel
    
    @State private var scrollToBottomId: UUID? = nil
    
    var body: some View {
        VStack(spacing: 0) {
            // Corpus directory info
            HStack {
                Image(systemName: "folder")
                    .foregroundColor(.secondary)
                Text("Corpus: \(viewModel.selectedConversation?.corpusDir ?? "Not set")")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                    .padding(.vertical, 2)
                Spacer()
                Button(action: { viewModel.showingCorpusDialog = true }) {
                    Image(systemName: "pencil")
                        .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(Color(.systemGray))
            
            // Error message banner
            if let errorMessage = viewModel.errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.white)
                    Text(errorMessage)
                        .foregroundColor(.white)
                    Spacer()
                    Button(action: { viewModel.errorMessage = nil }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.white)
                    }
                }
                .padding()
                .background(Color.red)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
            
            // Messages
            ScrollViewReader { scrollView in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(conversation.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                                .transition(.opacity)
                        }
                        
                        // Invisible view at the bottom to scroll to
                        Color.clear
                            .frame(height: 1)
                            .id("BOTTOM")
                    }
                    .padding()
                }
                .onAppear {
                    scrollView.scrollTo("BOTTOM", anchor: .bottom)
                }
                .onChange(of: conversation.messages.count) { _ in
                    withAnimation {
                        scrollView.scrollTo("BOTTOM", anchor: .bottom)
                    }
                }
            }
            
            // Input area with loading indicator
            VStack(spacing: 0) {
                if viewModel.isLoading {
                    HStack {
                        ProgressView()
                            .padding(.leading, 8)
                        Text("Generating response...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                    }
                    .padding(.bottom, 4)
                }
                
                HStack {
                    TextField("Type a message...", text: $viewModel.newMessageText, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(1...5)
                    
                    Button(action: viewModel.sendMessage) {
                        if viewModel.isLoading {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle())
                        } else {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.title2)
                        }
                    }
                    .disabled(viewModel.newMessageText.isEmpty || viewModel.isLoading)
                }
                .padding()
                .background(Color(NSColor.windowBackgroundColor).shadow(radius: 0.5))
            }
        }
        .navigationTitle(conversation.title)
        .animation(.easeInOut, value: viewModel.isLoading)
        .animation(.easeInOut, value: viewModel.errorMessage != nil)
        .sheet(isPresented: $viewModel.showingCorpusDialog) {
            CorpusDirectoryDialog(viewModel: viewModel)
        }
    }
}

struct MessageBubble: View {
    let message: Message
    
    private var bubbleColor: Color {
        message.sender == .user ? .blue : Color(NSColor.controlBackgroundColor)
    }
    
    private var textColor: Color {
        message.sender == .user ? .white : .primary
    }
    
    private var alignment: HorizontalAlignment {
        message.sender == .user ? .trailing : .leading
    }
    
    var body: some View {
        HStack {
            if message.sender == .assistant {
                Spacer()
            }
            
            Text(message.content)
                .padding(12)
                .background(bubbleColor)
                .foregroundColor(textColor)
                .cornerRadius(16)
                .contextMenu {
                    Button(action: {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(message.content, forType: .string)
                    }) {
                        Label("Copy", systemImage: "doc.on.doc")
                    }
                }
            
            if message.sender == .user {
                Spacer()
            }
        }
        .padding(.horizontal, 8)
    }
}

// Preview
struct ChatView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            ChatView(
                conversation: ConversationData(
                    title: "Preview",
                    corpusDir: "~/Downloads",
                    messages: [
                        Message(content: "Hello, how can I help you today?", sender: .assistant),
                        Message(content: "I need help with my project", sender: .user)
                    ]
                ),
                viewModel: ChatViewModel()
            )
        }
    }
}
