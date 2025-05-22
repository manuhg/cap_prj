import SwiftUI

struct CorpusDirectoryDialog: View {
    @ObservedObject var viewModel: ChatViewModel
    @State private var tempPath: String
    @Environment(\.dismiss) private var dismiss
    
    init(viewModel: ChatViewModel) {
        self.viewModel = viewModel
        self._tempPath = State(initialValue: viewModel.corpusDir)
    }
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Corpus Directory")) {
                    TextField("Path to corpus directory", text: $tempPath)
                        .textFieldStyle(.roundedBorder)
                        .autocorrectionDisabled()
                }
                
                Section {
                    Button("Save") {
                        viewModel.updateCorpusDirectory(tempPath)
                        dismiss()
                    }
                    .frame(maxWidth: .infinity)
                    .buttonStyle(.borderedProminent)
                }
            }
            .navigationTitle("Corpus Directory")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }
} 
