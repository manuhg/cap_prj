import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = ChatViewModel()
    
    var body: some View {
        NavigationSplitView {
            // Sidebar
            List(selection: $viewModel.selectedConversation) {
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
            .navigationTitle("Conversations")
            .toolbar(id: "sidebar") {
                ToolbarItem(id: "new", placement: .navigation) {
                    Button(action: viewModel.createNewConversation) {
                        Label("New Conversation", systemImage: "plus")
                    }
                }
            }
        } detail: {
            if let conversation = viewModel.selectedConversation {
                ChatView(conversation: conversation, viewModel: viewModel)
            } else {
                Text("Select a conversation")
                    .foregroundColor(.secondary)
            }
        }
    }
}
