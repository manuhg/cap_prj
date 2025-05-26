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
            
            // Use a VStack to contain the message content
            VStack {
                // Use a custom Text view that handles URLs properly
                LinkifiedText(text: message.content)
            }
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
            
            if message.sender != .user {
                Spacer(minLength: 100)
            }
        }
        .padding(.horizontal, 8)
    }
}

// Custom Text view that properly handles file:// URLs
struct LinkifiedText: View {
    let text: String
    
    var body: some View {
        let detector = try! NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue)
        let matches = detector.matches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count))
        
        if matches.isEmpty {
            // No links found, just use regular Text
            Text(text)
        } else {
            // Links found, create a custom view
            LinkTextView(text: text, matches: matches)
        }
    }
}

struct LinkTextView: View {
    let text: String
    let matches: [NSTextCheckingResult]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(0..<processedText().count, id: \.self) { index in
                let item = processedText()[index]
                if item.isLink {
                    Button(action: {
                        if let url = URL(string: item.text) {
                            handleURL(url)
                        }
                    }) {
                        Text(item.displayText)
                            .foregroundColor(.blue)
                            .underline()
                    }
                    .buttonStyle(PlainButtonStyle())
                } else {
                    Text(item.text)
                }
            }
        }
    }
    
    private func processedText() -> [(text: String, displayText: String, isLink: Bool)] {
        var result: [(text: String, displayText: String, isLink: Bool)] = []
        var currentIndex = text.startIndex
        
        // Sort matches by range location
        let sortedMatches = matches.sorted { $0.range.location < $1.range.location }
        
        for match in sortedMatches {
            // Add text before the link
            let linkStartIndex = text.index(text.startIndex, offsetBy: match.range.location)
            if currentIndex < linkStartIndex {
                let textSegment = String(text[currentIndex..<linkStartIndex])
                result.append((textSegment, textSegment, false))
            }
            
            // Add the link
            let linkEndIndex = text.index(linkStartIndex, offsetBy: match.range.length)
            let linkText = String(text[linkStartIndex..<linkEndIndex])
            
            // Create display text (for file URLs, show just the filename)
            var displayText = linkText
            if let url = URL(string: linkText), url.scheme == "file" {
                displayText = url.lastPathComponent
            }
            
            result.append((linkText, displayText, true))
            currentIndex = linkEndIndex
        }
        
        // Add any remaining text after the last link
        if currentIndex < text.endIndex {
            let textSegment = String(text[currentIndex..<text.endIndex])
            result.append((textSegment, textSegment, false))
        }
        
        return result
    }
    
    private func handleURL(_ url: URL) {
        // Handle file:// URLs
        if url.scheme == "file" {
            // Create a clean file URL
            var fileURL = url
            
            // Extract page number from fragment if present
            var pageNumber: Int? = nil
            if let fragment = url.fragment, fragment.hasPrefix("page=") {
                let pageStr = fragment.dropFirst(5) // Remove "page="
                pageNumber = Int(pageStr)
                
                // Remove fragment from URL for clean file access
                if let urlWithoutFragment = URL(string: url.absoluteString.components(separatedBy: "#")[0]) {
                    fileURL = urlWithoutFragment
                }
            }
            
            // For PDF files with page numbers, create a special URL that Preview.app can understand
            if fileURL.pathExtension.lowercased() == "pdf", let pageNumber = pageNumber {
                // Create a URL with the Preview-specific page parameter
                let previewPagedURL = URL(string: "file://" + fileURL.path + "#page=" + String(pageNumber))
                if let previewPagedURL = previewPagedURL {
                    // Try to open with NSWorkspace which will use the default PDF viewer (usually Preview)
                    NSWorkspace.shared.open(previewPagedURL)
                    return
                }
            }
            
            // Fallback to regular file opening if not a PDF or no page number
            NSWorkspace.shared.open(fileURL)
        }
        // Handle localhost URLs
        else if url.host == "localhost" && url.path == "/pdf" {
            // Extract the file path and page number from the URL query parameters
            guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
                  let pathItem = components.queryItems?.first(where: { $0.name == "path" }),
                  let filePath = pathItem.value?.removingPercentEncoding else {
                return
            }
            
            // Create a file URL with the path
            let fileURL = URL(fileURLWithPath: filePath)
            
            // Open the file with NSWorkspace
            NSWorkspace.shared.open(fileURL)
        }
        // Handle other URLs
        else {
            NSWorkspace.shared.open(url)
        }
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
