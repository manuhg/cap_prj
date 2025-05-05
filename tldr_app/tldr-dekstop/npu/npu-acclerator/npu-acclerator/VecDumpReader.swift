import Foundation
import CoreML

/// A Swift reader for the vector dump files created by the C++ library.
/// Uses memory mapping to efficiently access the embedding vectors without data copying.
public class VecDumpReader {
    
    // MARK: - Header Structure
    /// Swift representation of the C++ VectorCacheDumpHeader structure
    struct VectorCacheDumpHeader {
        let numEntries: UInt32      // Number of embedding vectors/hashes
        let hashSizeBytes: UInt32   // Size of each hash in bytes
        let vectorSizeBytes: UInt32 // Size of each embedding vector in bytes
        let vectorDimensions: UInt32 // Number of dimensions in each vector
        
        init(data: UnsafeRawPointer) {
            // Read the header values from the raw memory
            numEntries = data.load(fromByteOffset: 0, as: UInt32.self)
            hashSizeBytes = data.load(fromByteOffset: 4, as: UInt32.self)
            vectorSizeBytes = data.load(fromByteOffset: 8, as: UInt32.self)
            vectorDimensions = data.load(fromByteOffset: 12, as: UInt32.self)
        }
    }
    
    // MARK: - Properties
    private var fileHandle: FileHandle?
    private var mappedData: UnsafeMutableRawPointer?
    private var mappedLength: Int = 0
    private var header: VectorCacheDumpHeader?
    private var vectorsBasePtr: UnsafePointer<Float>?
    private var hashesBasePtr: UnsafePointer<UInt64>?
    
    // MARK: - Initialization
    public init() {}
    
    deinit {
        close()
    }
    
    /// Opens and memory maps the vector dump file
    /// - Parameter filePath: Path to the vector dump file
    /// - Returns: True if mapping was successful
    public func open(filePath: String) -> Bool {
        // Make sure we're starting fresh
        close()
        
        do {
            // Open the file
            fileHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: filePath))
            
            // Get file size
            let fileSize = try fileHandle?.seekToEnd() ?? 0
            try fileHandle?.seek(toOffset: 0)
            
            guard fileSize > 0 else {
                print("Error: Empty file")
                return false
            }
            
            // Memory map the file
            let fileDescriptor = fileHandle?.fileDescriptor ?? -1
            mappedData = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fileDescriptor, 0)
            mappedLength = Int(fileSize)
            
            guard mappedData != MAP_FAILED else {
                print("Error: Failed to memory map file")
                return false
            }
            
            // Read the header
            header = VectorCacheDumpHeader(data: mappedData!)
            
            guard let header = header else {
                print("Error: Failed to read header")
                return false
            }
            
            // Calculate offsets
            let headerSize = 16 // Size of the header (4 UInt32 values)
            let vectorsSectionSize = Int(header.numEntries) * Int(header.vectorSizeBytes)
            
            // Set up pointers to the vectors and hashes sections
            vectorsBasePtr = mappedData!.advanced(by: headerSize).assumingMemoryBound(to: Float.self)
            hashesBasePtr = mappedData!.advanced(by: headerSize + vectorsSectionSize).assumingMemoryBound(to: UInt64.self)
            
            return true
        } catch {
            print("Error opening vector dump file: \(error)")
            close()
            return false
        }
    }
    
    /// Closes the file and unmaps the memory
    public func close() {
        if let mappedData = mappedData, mappedLength > 0 {
            munmap(mappedData, mappedLength)
            self.mappedData = nil
            mappedLength = 0
        }
        
        if let fileHandle = fileHandle {
            try? fileHandle.close()
            self.fileHandle = nil
        }
        
        header = nil
        vectorsBasePtr = nil
        hashesBasePtr = nil
    }
    
    /// Gets the number of vectors in the file
    public var count: Int {
        return Int(header?.numEntries ?? 0)
    }
    
    /// Gets the dimensions of each vector
    public var dimensions: Int {
        return Int(header?.vectorDimensions ?? 0)
    }
    
    /// Gets the hash for a specific vector at index
    /// - Parameter index: Index of the vector
    /// - Returns: The hash value or nil if index is out of bounds
    public func getHash(at index: Int) -> UInt64? {
        guard let header = header, let hashesBasePtr = hashesBasePtr else { return nil }
        guard index >= 0 && index < header.numEntries else { return nil }
        
        return hashesBasePtr.advanced(by: index).pointee
    }
    
    /// Gets a vector at a specific index as a direct pointer
    /// - Parameter index: Index of the vector
    /// - Returns: Pointer to the vector data or nil if index is out of bounds
    public func getVectorPointer(at index: Int) -> UnsafePointer<Float>? {
        guard let header = header, let vectorsBasePtr = vectorsBasePtr else { return nil }
        guard index >= 0 && index < header.numEntries else { return nil }
        
        return vectorsBasePtr.advanced(by: index * Int(header.vectorDimensions))
    }
    
    /// Creates an MLMultiArray directly from memory-mapped vector data without copying
    /// - Parameter index: Index of the vector
    /// - Returns: MLMultiArray containing the vector or nil if creation failed
    public func getVectorAsMLMultiArray(at index: Int) -> MLMultiArray? {
        guard let header = header, 
              let vectorPointer = getVectorPointer(at: index),
              index >= 0 && index < header.numEntries else { return nil }
        
        do {
            // Create a multi array with the same dimensions as the vector
            let shape = [NSNumber(value: header.vectorDimensions)]
            let multiArray = try MLMultiArray(dataPointer: UnsafeMutableRawPointer(mutating: vectorPointer),
                                             shape: shape,
                                             dataType: .float32,
                                             strides: [NSNumber(value: 1)],
                                             deallocator: nil) // No deallocator as we're not allocating memory
            
            return multiArray
        } catch {
            print("Error creating MLMultiArray: \(error)")
            return nil
        }
    }
    
    /// Creates an MLMultiArray directly from all memory-mapped vectors without copying
    /// - Returns: MLMultiArray containing all vectors or nil if creation failed
    public func getAllVectorsAsMLMultiArray() -> MLMultiArray? {
        guard let header = header, let vectorsBasePtr = vectorsBasePtr else { return nil }
        
        do {
            // Create a multi array with all vectors (numEntries × vectorDimensions)
            let shape = [NSNumber(value: header.numEntries), NSNumber(value: header.vectorDimensions)]
            let multiArray = try MLMultiArray(dataPointer: UnsafeMutableRawPointer(mutating: vectorsBasePtr),
                                             shape: shape, 
                                             dataType: .float32,
                                             strides: [NSNumber(value: header.vectorDimensions), NSNumber(value: 1)],
                                             deallocator: nil) // No deallocator as we're not allocating memory
            
            return multiArray
        } catch {
            print("Error creating MLMultiArray: \(error)")
            return nil
        }
    }
    
    /// Prints information about the vector dump file
    public func printInfo() {
        guard let header = header else {
            print("No header information available")
            return
        }
        
        print("=== Vector Cache File Info ===")
        print("Number of entries: \(header.numEntries)")
        print("Hash size (bytes): \(header.hashSizeBytes)")
        print("Vector size (bytes): \(header.vectorSizeBytes)")
        print("Vector dimensions: \(header.vectorDimensions)")
        
        // Print sample entry (index 1) if available
        if header.numEntries > 1, let hash = getHash(at: 1), let vectorPtr = getVectorPointer(at: 1) {
            print("\nSample element (index 1):")
            print("Hash: \(hash)")
            
            print("Embedding vector (first 10 dimensions):")
            let dimsToShow = min(10, Int(header.vectorDimensions))
            for i in 0..<dimsToShow {
                print(vectorPtr.advanced(by: i).pointee, terminator: i < dimsToShow - 1 ? ", " : "")
            }
            print(header.vectorDimensions > 10 ? "..." : "")
        }
    }
}

// MARK: - Usage Example

/* Example usage:
 
let reader = VecDumpReader()
if reader.open(filePath: "/path/to/vector_cache_file.bin") {
    reader.printInfo()
    
    // Get a single vector as MLMultiArray
    if let vector = reader.getVectorAsMLMultiArray(at: 1) {
        print("Successfully created MLMultiArray for vector at index 1")
        // Use vector with Core ML model
    }
    
    // Get all vectors as one MLMultiArray
    if let allVectors = reader.getAllVectorsAsMLMultiArray() {
        print("Successfully created MLMultiArray with all \(reader.count) vectors")
        // Use allVectors with Core ML model for batch processing
    }
    
    reader.close()
}
 
*/
