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
                            MessageBubble(message: message, viewModel: viewModel)
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
            
            // Resizable info panel
            VStack(spacing: 0) {
                if viewModel.infoText != nil {
                    ScrollView {
                        Text(viewModel.infoText ?? "")
                            .font(.system(.caption, design: .monospaced))
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled) // Make text selectable
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: viewModel.infoPanelHeight)
                    .background(Color(.textBackgroundColor))
                    .overlay(
                        Rectangle()
                            .frame(height: 1)
                            .foregroundColor(Color(.separatorColor)),
                        alignment: .bottom
                    )
                    // Add a resize handle at the bottom
                    .overlay(
                        HStack {
                            Spacer()
                            Image(systemName: "arrow.up.and.down")
                                .foregroundColor(.secondary)
                                .padding(4)
                                .background(Color.secondary.opacity(0.1))
                                .cornerRadius(4)
                                .contentShape(Rectangle())
                                .gesture(
                                    DragGesture()
                                        .onChanged { value in
                                            let newHeight = viewModel.infoPanelHeight - value.translation.height
                                            // Maintain the increased max height of 600 for more resizing range
                                            viewModel.infoPanelHeight = max(80, min(600, newHeight))
                                        }
                                )
                        }
                        .padding(.trailing, 4),
                        alignment: .bottom
                    )
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
        }
        .navigationTitle(conversation.title)
        .animation(.easeInOut, value: viewModel.isLoading)
        .animation(.easeInOut, value: viewModel.errorMessage != nil)
    }
}

struct MessageBubble: View {
    let message: Message
    let viewModel: ChatViewModel
    
    private var bubbleColor: Color {
        switch message.sender {
        case .user:
            return .blue
        case .system:
            return Color(.darkGray)
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
                
                // Show a small indicator if the message has source context
                if message.hasSourceContext {
                    HStack {
                        Spacer()
                        Label("Context available", systemImage: "doc.text.magnifyingglass")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.top, 4)
                    }
                }
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
                
                // Add option to view sources if available
                if message.hasSourceContext {
                    Button(action: {
                        viewModel.showSourcesInInfoPanel(for: message)
                    }) {
                        Label("View Sources", systemImage: "doc.text.magnifyingglass")
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

// Custom Text view that properly handles file:// URLs
struct LinkifiedText: View {
    let text: String
    
    var body: some View {
        // Use a custom URL detection approach that better handles file URLs with fragments
        let urls = extractURLs(from: text)
        
        if urls.isEmpty {
            // No URLs found, just use regular Text
            Text(text)
                .textSelection(.enabled) // Make text selectable
        } else {
            // URLs found, create a custom view with our extracted URLs
            CustomLinkTextView(text: text, urls: urls)
        }
    }
    
    // Custom URL extraction that handles file:// URLs with page numbers
    private func extractURLs(from text: String) -> [(range: Range<String.Index>, url: URL)] {
        var results: [(range: Range<String.Index>, url: URL)] = []
        
        // First try the standard NSDataDetector
        let detector = try! NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue)
        let matches = detector.matches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count))
        
        for match in matches {
            if let range = Range(match.range, in: text), let url = URL(string: String(text[range])) {
                results.append((range, url))
            }
        }
        
        // Then look for file:// URLs that might not be detected properly
        let filePrefix = "file://"
        var searchRange = text.startIndex..<text.endIndex
        
        while let fileStartRange = text.range(of: filePrefix, options: [], range: searchRange) {
            // Find the end of the URL (whitespace or newline)
            let urlStartIndex = fileStartRange.lowerBound
            var urlEndIndex = text.endIndex
            
            // Use indices to safely iterate through String.Index values
            var currentIndex = fileStartRange.upperBound
            while currentIndex < text.endIndex {
                let char = text[currentIndex]
                if char.isWhitespace || char.isNewline {
                    urlEndIndex = currentIndex
                    break
                }
                currentIndex = text.index(after: currentIndex)
            }
            
            let urlRange = urlStartIndex..<urlEndIndex
            let urlString = String(text[urlRange])
            
            // Only add if it's a valid URL and not already in our results
            if let url = URL(string: urlString),
               !results.contains(where: { $0.range.overlaps(urlRange) }) {
                results.append((urlRange, url))
            }
            
            // Update search range for next iteration
            searchRange = urlEndIndex..<text.endIndex
        }
        
        // Sort by range location
        return results.sorted { $0.range.lowerBound < $1.range.lowerBound }
    }
}

struct CustomLinkTextView: View {
    let text: String
    let urls: [(range: Range<String.Index>, url: URL)]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(0..<processedText().count, id: \.self) { index in
                let item = processedText()[index]
                if item.isLink {
                    Button(action: {
                        if let url = item.url {
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
                        .textSelection(.enabled) // Make text selectable
                }
            }
        }
    }
    
    private func processedText() -> [(text: String, displayText: String, isLink: Bool, url: URL?)] {
        var result: [(text: String, displayText: String, isLink: Bool, url: URL?)] = []
        var currentIndex = text.startIndex
        
        for urlInfo in urls {
            // Add text before the link
            if currentIndex < urlInfo.range.lowerBound {
                let textSegment = String(text[currentIndex..<urlInfo.range.lowerBound])
                result.append((textSegment, textSegment, false, nil))
            }
            
            // Add the link
            let linkText = String(text[urlInfo.range])
            
            // Create display text (for file URLs, show just the filename)
            var displayText = linkText
            if urlInfo.url.scheme == "file" {
                // For file URLs, just show the filename
                displayText = urlInfo.url.lastPathComponent
                
                // If there's a page number, add it to the display
                if let fragment = urlInfo.url.fragment, fragment.hasPrefix("page="),
                   let pageNumber = Int(fragment.dropFirst(5)) {
                    displayText += " (page \(pageNumber))"
                }
            }
            
            result.append((linkText, displayText, true, urlInfo.url))
            currentIndex = urlInfo.range.upperBound
        }
        
        // Add any remaining text after the last link
        if currentIndex < text.endIndex {
            let textSegment = String(text[currentIndex..<text.endIndex])
            result.append((textSegment, textSegment, false, nil))
        }
        
        return result
    }
    
    private func handleURL(_ url: URL) {
        print("Opening URL: \(url)")
        
        // Handle file:// URLs - assuming all are PDF files with page numbers
        if url.scheme == "file" {
            // Use AppleScript to open the URL with page number intact
            print("Opening PDF with AppleScript: \(url)")
            
            // Create AppleScript to open the URL in Brave Browser
            let script = "tell application \"Brave Browser\" to open location \"\(url.absoluteString)\""
            
            // Execute the AppleScript
            var error: NSDictionary?
            if let scriptObject = NSAppleScript(source: script) {
                scriptObject.executeAndReturnError(&error)
                
                if let error = error {
                    print("Error executing AppleScript: \(error)")
                    // Fallback to regular open
                    NSWorkspace.shared.open(url)
                }
            } else {
                // Fallback to regular open if script creation fails
                NSWorkspace.shared.open(url)
            }
            return
        }
        // Handle localhost URLs
        else if url.host == "localhost" && url.path == "/pdf" {
            // Extract the file path and page number from the URL query parameters
            guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
                  let pathItem = components.queryItems?.first(where: { $0.name == "path" }),
                  let filePath = pathItem.value?.removingPercentEncoding else {
                return
            }
            
            // Get the page number if available
            let pageNumber = components.queryItems?.first(where: { $0.name == "page" })?.value
            
            // Create a file URL with the path
            let fileURL = URL(fileURLWithPath: filePath)
            
            // Create a URL with page number if available
            var urlString = "file://" + filePath
            if let pageNumber = pageNumber {
                urlString += "#page=" + pageNumber
            }
            
            // Use AppleScript to open the URL in Brave Browser
            let script = "tell application \"Brave Browser\" to open location \"\(urlString)\""
            
            // Execute the AppleScript
            var error: NSDictionary?
            if let scriptObject = NSAppleScript(source: script) {
                scriptObject.executeAndReturnError(&error)
                
                if let error = error {
                    print("Error executing AppleScript: \(error)")
                    // Fallback to regular open
                    NSWorkspace.shared.open(fileURL)
                }
            } else {
                // Fallback to regular open if script creation fails
                NSWorkspace.shared.open(fileURL)
            }
        }
        // Handle other URLs
        else {
            // Use AppleScript to open the URL in Brave Browser
            let script = "tell application \"Brave Browser\" to open location \"\(url.absoluteString)\""
            
            // Execute the AppleScript
            var error: NSDictionary?
            if let scriptObject = NSAppleScript(source: script) {
                scriptObject.executeAndReturnError(&error)
                
                if let error = error {
                    print("Error executing AppleScript: \(error)")
                    // Fallback to regular open
                    NSWorkspace.shared.open(url)
                }
            } else {
                // Fallback to regular open if script creation fails
                NSWorkspace.shared.open(url)
            }
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
