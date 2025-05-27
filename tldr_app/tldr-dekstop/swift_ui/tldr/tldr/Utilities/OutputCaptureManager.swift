import Foundation

/// A class that captures stdout and stderr output and forwards it to a callback function
class OutputCaptureManager {
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var originalStdout: Int32?
    private var originalStderr: Int32?
    private var outputCallback: ((String, Bool) -> Void)?
    
    /// Start capturing stdout and stderr
    /// - Parameter callback: A function that will be called with the captured output. 
    ///   The first parameter is the output text, the second parameter is true for stderr, false for stdout.
    func startCapturing(callback: @escaping (String, Bool) -> Void) {
        outputCallback = callback
        
        // Save original file descriptors
        originalStdout = dup(FileHandle.standardOutput.fileDescriptor)
        originalStderr = dup(FileHandle.standardError.fileDescriptor)
        
        // Create pipes
        stdoutPipe = Pipe()
        stderrPipe = Pipe()
        
        // Redirect stdout
        dup2(stdoutPipe!.fileHandleForWriting.fileDescriptor, FileHandle.standardOutput.fileDescriptor)
        
        // Redirect stderr
        dup2(stderrPipe!.fileHandleForWriting.fileDescriptor, FileHandle.standardError.fileDescriptor)
        
        // Set up async reading from stdout pipe
        stdoutPipe?.fileHandleForReading.readabilityHandler = { [weak self] fileHandle in
            guard let self = self, let callback = self.outputCallback else { return }
            
            let data = fileHandle.availableData
            if let output = String(data: data, encoding: .utf8) {
                callback(output, false) // false for stdout
            }
        }
        
        // Set up async reading from stderr pipe
        stderrPipe?.fileHandleForReading.readabilityHandler = { [weak self] fileHandle in
            guard let self = self, let callback = self.outputCallback else { return }
            
            let data = fileHandle.availableData
            if let output = String(data: data, encoding: .utf8) {
                callback(output, true) // true for stderr
            }
        }
    }
    
    /// Stop capturing stdout and stderr
    func stopCapturing() {
        // Restore original stdout
        if let originalStdout = originalStdout {
            dup2(originalStdout, FileHandle.standardOutput.fileDescriptor)
            close(originalStdout)
            self.originalStdout = nil
        }
        
        // Restore original stderr
        if let originalStderr = originalStderr {
            dup2(originalStderr, FileHandle.standardError.fileDescriptor)
            close(originalStderr)
            self.originalStderr = nil
        }
        
        // Remove readability handlers
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        
        // Close pipes
        try? stdoutPipe?.fileHandleForWriting.close()
        try? stderrPipe?.fileHandleForWriting.close()
        
        stdoutPipe = nil
        stderrPipe = nil
        outputCallback = nil
    }
}
