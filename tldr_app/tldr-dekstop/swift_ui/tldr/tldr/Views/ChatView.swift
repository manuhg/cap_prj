import SwiftUI
@MainActor
struct ChatView: View {
    @ObservedObject var viewModel: ChatViewModel
    
    // Use a computed property to get the current conversation from the viewModel
    private var conversation: ConversationData {
        viewModel.selectedConversation ?? ConversationData(title: "No Conversation")
    }
    
    @State private var scrollToBottomId: UUID? = nil
    
    var body: some View {
        VStack(spacing: 0) {
            // Corpus directory info
            HStack {
                Image(systemName: "folder")
                    .foregroundColor(.secondary)
                Text("Corpus: \(viewModel.selectedConversation?.corpusDir ?? "Not set")")
                    .font(.headline)
                    .foregroundColor(.black)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                    .padding(.vertical, 2)
                Spacer()
                Button(action: viewModel.selectCorpusDirectory) {
                    Image(systemName: "pencil")
                        .foregroundColor(.black)
                }
                Button(action: viewModel.addCorpus) {
                    Text("Analyze")
                        .foregroundColor(.black)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .cornerRadius(4)
                }
                .disabled(viewModel.isLoading)
                .help("Analyze corpus directory to generate embeddings")
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
                .onChange(of: viewModel.selectedConversation?.id) { _ in
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
                        .onSubmit {
                            if !viewModel.newMessageText.isEmpty && !viewModel.isLoading {
                                viewModel.sendMessage()
                            }
                        }
                        .submitLabel(.send)
                    
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
    }
}

struct MessageBubble: View {
    let message: Message
    
    private var bubbleColor: Color {
        switch message.sender {
        case .user:
            return .blue
        case .system:
            return Color(.systemGreen)
        case .assistant:
            return Color(NSColor.controlBackgroundColor)
        }
    }
    
    private var textColor: Color {
        switch message.sender {
        case .user, .system:
            return .white
        case .assistant:
            return .primary
        }
    }
    
    private var alignment: Alignment {
        switch message.sender {
        case .user:
            return .trailing
        case .system, .assistant:
            return .leading
        }
    }
    
    var body: some View {
        HStack {
            if message.sender == .user {
                Spacer(minLength: 100)
            }
            
            // Use NSTextView for better link handling
            if message.content.contains("file://") {
                ClickableLinksTextView(text: message.content, textColor: textColor)
                    .padding(12)
                    .background(bubbleColor)
                    .cornerRadius(16)
                    .frame(maxWidth: 600, alignment: alignment)
                    .contextMenu {
                        Button(action: {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(message.content, forType: .string)
                        }) {
                            Label("Copy", systemImage: "doc.on.doc")
                        }
                    }
            } else {
                // Regular text for messages without links
                Text(message.content)
                    .padding(12)
                    .background(bubbleColor)
                    .foregroundColor(textColor)
                    .cornerRadius(16)
                    .frame(maxWidth: 600, alignment: alignment)
                    .contextMenu {
                        Button(action: {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(message.content, forType: .string)
                        }) {
                            Label("Copy", systemImage: "doc.on.doc")
                        }
                    }
            }
            
            if message.sender != .user {
                Spacer(minLength: 100)
            }
        }
        .padding(.horizontal, 8)
    }
}

// NSTextView wrapper for clickable links
struct ClickableLinksTextView: NSViewRepresentable {
    let text: String
    let textColor: Color
    
    func makeNSView(context: Context) -> NSScrollView {
        // Create a text storage with the attributed string
        let textStorage = NSTextStorage(attributedString: createAttributedString())
        
        // Create layout manager and text container
        let layoutManager = NSLayoutManager()
        textStorage.addLayoutManager(layoutManager)
        
        let textContainer = NSTextContainer(containerSize: CGSize(width: 600, height: CGFloat.greatestFiniteMagnitude))
        textContainer.widthTracksTextView = true
        layoutManager.addTextContainer(textContainer)
        
        // Create NSTextView with the text container
        let textView = NSTextView(frame: .zero, textContainer: textContainer)
        textView.isEditable = false
        textView.isSelectable = true
        textView.drawsBackground = false
        textView.textColor = NSColor(textColor)
        textView.font = NSFont.systemFont(ofSize: NSFont.systemFontSize)
        textView.isAutomaticLinkDetectionEnabled = true
        textView.allowsUndo = false
        textView.textContainerInset = NSSize(width: 0, height: 0)
        
        // Create a scroll view to contain the text view
        let scrollView = NSScrollView()
        scrollView.documentView = textView
        scrollView.hasVerticalScroller = false
        scrollView.hasHorizontalScroller = false
        scrollView.drawsBackground = false
        
        // Size the text view to fit its content
        textView.sizeToFit()
        
        return scrollView
    }
    
    func updateNSView(_ nsView: NSScrollView, context: Context) {
        if let textView = nsView.documentView as? NSTextView {
            textView.textStorage?.setAttributedString(createAttributedString())
            textView.textColor = NSColor(textColor)
            textView.sizeToFit()
        }
    }
    
    private func createAttributedString() -> NSAttributedString {
        let attributedString = NSMutableAttributedString(string: text)
        
        // Find file:/// URLs using regex
        let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue)
        let range = NSRange(location: 0, length: text.utf16.count)
        
        // Apply link attributes to URLs
        detector?.enumerateMatches(in: text, options: [], range: range) { (match, _, _) in
            if let match = match, let url = match.url {
                attributedString.addAttribute(.link, value: url, range: match.range)
                attributedString.addAttribute(.foregroundColor, value: NSColor.blue, range: match.range)
                attributedString.addAttribute(.underlineStyle, value: NSUnderlineStyle.single.rawValue, range: match.range)
            }
        }
        
        // Additional regex for file:/// URLs which might not be detected by NSDataDetector
        let fileURLPattern = "file://[^ \n]+"
        if let regex = try? NSRegularExpression(pattern: fileURLPattern, options: []) {
            regex.enumerateMatches(in: text, options: [], range: range) { (match, _, _) in
                if let match = match {
                    let urlString = (text as NSString).substring(with: match.range)
                    if let url = URL(string: urlString) {
                        attributedString.addAttribute(.link, value: url, range: match.range)
                        attributedString.addAttribute(.foregroundColor, value: NSColor.blue, range: match.range)
                        attributedString.addAttribute(.underlineStyle, value: NSUnderlineStyle.single.rawValue, range: match.range)
                    }
                }
            }
        }
        
        return attributedString
    }
}

// Preview
struct ChatView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            let viewModel = ChatViewModel()
            // Set up the preview viewModel with a test conversation
            let previewConversation = ConversationData(
                title: "Preview",
                corpusDir: "~/Downloads",
                messages: [
                    Message(content: "Hello, how can I help you today?", sender: .assistant),
                    Message(content: "I need help with my project", sender: .user)
                ]
            )
            viewModel.conversations = [previewConversation]
            viewModel.selectedConversation = previewConversation
            
            return ChatView(viewModel: viewModel)
        }
    }
}
