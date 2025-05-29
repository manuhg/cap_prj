import SwiftUI
@MainActor
struct ContentView: View {
    @StateObject private var viewModel = ChatViewModel()
    @State private var showingDeleteConfirmation = false
    @State private var conversationToDelete: ConversationData?
    
    var body: some View {
        NavigationSplitView {
            // Sidebar
            VStack(spacing: 0) {
                // TLDR branding
                HStack(spacing: 12) {
                    if let nsImage = NSImage(named: NSImage.applicationIconName) {
                        Image(nsImage: nsImage)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 32, height: 32)
                            .background(Color.clear)
                            .drawingGroup(opaque: false)
                    }

                    Text("TLDR")
                        .font(.system(size: 24, weight: .bold))

                    Spacer()
                }
                .padding(.horizontal)
                .padding(.vertical, 12)
                
                Divider()
                
                List(selection: Binding(
    get: { viewModel.selectedConversation },
    set: { newValue in
        // If there's an active query, handle it properly before switching conversations
        if viewModel.isLoading {
            // First set isLoading to false to stop any UI updates related to the loading state
            viewModel.isLoading = false
        }
        viewModel.selectedConversation = newValue
    }
)) {
                ForEach(viewModel.conversations) { conversation in
                    NavigationLink(value: conversation) {
                        VStack(alignment: .leading) {
                            Text(conversation.title)
                                .font(.headline)
                            Text(conversation.lastUpdated, style: .date)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
                .safeAreaInset(edge: .bottom) {
                    if let selectedConversation = viewModel.selectedConversation {
                        VStack(spacing: 0) {
                            Divider()
                            HStack {
                                Button(action: {
                                    conversationToDelete = selectedConversation
                                    showingDeleteConfirmation = true
                                }) {
                                    Label("Delete Conversation", systemImage: "trash")
                                        .foregroundColor(.red)
                                }
                                .buttonStyle(.plain)
                                .padding(8)
                                Spacer()
                            }
                            .background(Color(.windowBackgroundColor))
                        }
                    }
                }
            }
            .navigationTitle("Conversations")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button(action: viewModel.createNewConversation) {
                        Label("New Conversation", systemImage: "plus")
                    }
                }
            }
        } detail: {
            if viewModel.selectedConversation != nil {
                ChatView(viewModel: viewModel)
            } else {
                Text("Select a conversation")
                    .foregroundColor(.secondary)
            }
        }
        .confirmationDialog(
            "Delete Conversation",
            isPresented: $showingDeleteConfirmation,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                if let conversation = conversationToDelete {
                    viewModel.deleteConversation(conversation)
                }
                conversationToDelete = nil
            }
            Button("Cancel", role: .cancel) {
                conversationToDelete = nil
            }
        } message: {
            Text("Are you sure you want to delete this conversation? This action cannot be undone.")
        }
    }
}
