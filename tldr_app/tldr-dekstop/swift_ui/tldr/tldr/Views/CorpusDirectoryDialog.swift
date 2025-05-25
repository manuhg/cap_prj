import SwiftUI
import UniformTypeIdentifiers

struct CorpusDirectoryDialog: View {
    @ObservedObject var viewModel: ChatViewModel
    @State private var tempPath: String
    @State private var isShowingDirectoryPicker = false
    @Environment(\.dismiss) private var dismiss
    
    init(viewModel: ChatViewModel) {
        self.viewModel = viewModel
        self._tempPath = State(initialValue: viewModel.corpusDir)
    }
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Corpus Directory")
                .font(.title2)
                .fontWeight(.bold)
                .padding(.top)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Directory Location")
                    .font(.headline)
                
                HStack {
                    TextField("Select a directory", text: $tempPath)
                        .textFieldStyle(.roundedBorder)
                        .disabled(true)
                        .foregroundColor(.secondary)
                    
                    Button(action: {
                        isShowingDirectoryPicker = true
                    }) {
                        Image(systemName: "folder")
                            .imageScale(.medium)
                            .padding(8)
                            .background(Color.blue.opacity(0.1))
                            .cornerRadius(8)
                    }
                    .buttonStyle(PlainButtonStyle())
                }
                
                Text("Select the directory containing your corpus files")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color.secondary.opacity(0.1))
            .cornerRadius(12)
            .padding(.horizontal)
            
            Spacer()
            
            HStack(spacing: 16) {
                Button("Cancel") {
                    dismiss()
                }
                .frame(maxWidth: .infinity)
                .frame(height: 44)
                .background(Color.gray.opacity(0.2))
                .foregroundColor(.primary)
                .cornerRadius(10)
                
                Button("Save") {
                    viewModel.updateCorpusDirectory(tempPath)
                    dismiss()
                }
                .frame(maxWidth: .infinity)
                .frame(height: 44)
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .padding(.horizontal)
            .padding(.bottom, 20)
        }
        .frame(width: 500, height: 300)
        .fileImporter(
            isPresented: $isShowingDirectoryPicker,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    tempPath = url.path(percentEncoded: false)
                }
            case .failure(let error):
                print("Error selecting directory: \(error.localizedDescription)")
            }
        }
    }
}

#Preview {
    struct PreviewWrapper: View {
        @StateObject var viewModel = ChatViewModel()
        
        var body: some View {
            CorpusDirectoryDialog(viewModel: viewModel)
        }
    }
    return PreviewWrapper()
}
