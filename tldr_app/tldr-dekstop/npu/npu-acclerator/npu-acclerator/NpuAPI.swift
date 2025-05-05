import Foundation
import CoreML

// MARK: - Constants

let VECTOR_DIM = 384
let BATCH_SIZE = 1024*128 // Default batch size (must be <= 1024)

// MARK: - Vector Similarity Result Types

/// Structure for returning similarity results to C++
struct SimilarityResult {
    var hash: UInt64 // Hash value of the vector
    var score: Float // Similarity score (higher is more similar)
}

/// Internal type for tracking vector similarity results during processing
typealias VectorSimilarityResult = (index: Int, score: Float, hash: UInt64)

// MARK: - Vector Creation Helpers

/// Creates an MLMultiArray with a specific shape and fills it with the given value
/// - Parameters:
///   - shape: The shape of the array to create
///   - value: The value to fill the array with
/// - Returns: An initialized MLMultiArray
func createVector(shape: [NSNumber], value: Float) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape, dataType: .float32)
    let count = array.count
    let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: count)
    for i in 0..<count {
        pointer[i] = value
    }
    return array
}

/// Creates an MLMultiArray with a specific shape and fills it with random values
/// - Parameter shape: The shape of the array to create
/// - Returns: An initialized MLMultiArray with random values
func createRandomVector(shape: [NSNumber]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape, dataType: .float32)
    let count = array.count
    let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: count)
    for i in 0..<count {
        pointer[i] = Float.random(in: -1.0...1.0) // Example random range
    }
    return array
}

// MARK: - C API Interface

/// Creates a C-compatible array from Swift similarity results
/// - Parameter results: Array of vector similarity results
/// - Returns: A pointer to C-compatible SimilarityResult array (must be freed by caller)
func createResultArray(_ results: [VectorSimilarityResult]) -> UnsafeMutablePointer<SimilarityResult> {
    let resultsPtr = UnsafeMutablePointer<SimilarityResult>.allocate(capacity: results.count)
    
    for i in 0..<results.count {
        resultsPtr[i] = SimilarityResult(hash: results[i].hash, score: results[i].score)
    }
    
    return resultsPtr
}

/// Loads the CosineSimilarityBatched model from the specified path
/// - Parameters:
///   - modelPath: Path to the compiled .mlmodelc directory
///   - useNeuralEngine: Whether to use the Neural Engine for inference
/// - Returns: The loaded ML model
func loadCosineSimilarityModel(from modelPath: String, useNeuralEngine: Bool = true) throws -> CosineSimilarityBatched {
    let configuration = MLModelConfiguration()
    if useNeuralEngine {
        configuration.computeUnits = .cpuAndNeuralEngine
    } else {
        configuration.computeUnits = .cpuOnly
    }
    
    print("Loading model from: \(modelPath)")
    let modelURL = URL(fileURLWithPath: modelPath)
    return try CosineSimilarityBatched(contentsOf: modelURL, configuration: configuration)
}

// MARK: - Vector Processing Helper Functions

/// Creates an MLMultiArray from a float pointer
/// - Parameters:
///   - pointer: Pointer to float data
///   - dimensions: Number of dimensions in the vector
/// - Returns: MLMultiArray containing the vector data
/// - Throws: If vector creation fails
func createVectorFromPointer(pointer: UnsafePointer<Float32>, dimensions: Int32) throws -> MLMultiArray {
    print("Creating vector from pointer, dimensions: \(dimensions)")
    
    // Create a new MLMultiArray with shape [1, dimensions] for the query vector
    let vector = try MLMultiArray(shape: [1, NSNumber(value: dimensions)], dataType: .float32)
    
    // Copy the data from the pointer to the MLMultiArray
    let vecPtr = vector.dataPointer.bindMemory(to: Float32.self, capacity: Int(dimensions))
    for i in 0..<Int(dimensions) {
        vecPtr[i] = pointer.advanced(by: i).pointee
    }
    
    return vector
}

/// Creates a query vector from the first vector in the dump file
/// - Parameters:
///   - reader: The VecDumpReader with access to vectors
/// - Returns: MLMultiArray containing the query vector
/// - Throws: If vector creation fails
func createQueryVector(from reader: VecDumpReader) throws -> MLMultiArray {
    print("Creating query vector from first vector in dump...")
    
    // Create a new MLMultiArray with shape [1, dimensions] for the query vector
    let queryVector = try MLMultiArray(shape: [1, NSNumber(value: reader.dimensions)], dataType: .float32)
    
    // Copy the first vector from vec_arr to queryVector
    guard let firstVectorPtr = reader.getVectorPointer(at: 0) else {
        throw NSError(domain: "NpuAPI", code: 1, 
                     userInfo: [NSLocalizedDescriptionKey: "Could not get first vector as query vector"])
    }
    
    let queryPtr = queryVector.dataPointer.bindMemory(to: Float32.self, capacity: Int(reader.dimensions))
    for i in 0..<Int(reader.dimensions) {
        queryPtr[i] = firstVectorPtr.advanced(by: i).pointee
    }
    
    print("Query vector prepared successfully, shape: \(queryVector.shape)")
    return queryVector
}

/// Computes cosine similarity between a query vector and a set of vectors
/// - Parameters:
///   - modelPath: Path to the CoreML model
///   - queryVector: Query vector as MLMultiArray
///   - vectors: Vectors to compare with as MLMultiArray
///   - hashes: Optional array of hash values corresponding to the vectors
/// - Returns: Array of top similarity results with index, score, and hash
/// - Throws: If similarity calculation fails
func computeCosineSimilarity(
    using model: CosineSimilarityBatched,
    queryVector: MLMultiArray,
    vectors: MLMultiArray,
    hashes: [UInt64]? = nil
) throws -> [VectorSimilarityResult] {
    print("Computing cosine similarity...")
    print("Query vector shape: \(queryVector.shape), Vectors shape: \(vectors.shape)")
    
    // Perform prediction
    let startTime = Date()
    let predictionOutput = try model.prediction(input1: queryVector, input2: vectors)
    let calculationTime = Date().timeIntervalSince(startTime) * 1000
    
    // Extract similarity scores
    guard let similarityScores = predictionOutput.featureValue(for: "var_5")?.multiArrayValue else {
        throw NSError(domain: "NpuAPI", code: 2, 
                     userInfo: [NSLocalizedDescriptionKey: "Could not extract similarity scores"])
    }
    
    print("Cosine similarity calculation time: \(calculationTime) milliseconds")
    print("Output similarities shape: \(similarityScores.shape)")
    
    // Process scores
    let scorePtr = similarityScores.dataPointer.bindMemory(to: Float32.self, capacity: similarityScores.count)
    var results: [(index: Int, score: Float, hash: UInt64)] = []
    
    for i in 0..<similarityScores.count {
        let score = scorePtr[i]
        let hash: UInt64 = hashes?[safe: i] ?? UInt64(i) // Use provided hash if available, otherwise use index
        
        // Skip first vector if it's the query vector
        if i == 0 && score > 0.99 { continue }
        
        results.append((i, score, hash))
    }
    
    // Sort by similarity score (descending)
    results.sort { $0.score > $1.score }
    
    return results
}

// Extension to safely access array elements
extension Array {
    subscript(safe index: Index) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}

/// Processes a batch of vectors for similarity calculation
/// - Parameters:
///   - batchIndex: Index of current batch
///   - batchCount: Total number of batches
///   - startIndex: Start index of vectors in this batch
///   - size: Size of this batch
///   - dimensions: Number of dimensions in each vector
///   - reader: VecDumpReader to access vectors
///   - model: ML model for similarity calculation
///   - queryVector: Query vector to compare against
///   - indexToHash: Dictionary to store hash mappings (modified in place)
/// - Returns: Dictionary of global indices to similarity scores
/// - Throws: If batch processing fails
func processBatch(
    index batchIndex: Int, 
    of batchCount: Int,
    startIndex: Int,
    size: Int,
    dimensions: UInt32,
    reader: VecDumpReader,
    model: CosineSimilarityBatched,
    queryVector: MLMultiArray,
    indexToHash: inout [Int: UInt64]
) throws -> [Int: Float] {
    print("\nProcessing batch \(batchIndex + 1)/\(batchCount): vectors \(startIndex) to \(startIndex + size - 1)")
    
    // Create batch MLMultiArray
    let batchVectors = try MLMultiArray(shape: [NSNumber(value: size), NSNumber(value: dimensions)], 
                                       dataType: .float32)
    
    // Fill batch with vectors
    for i in 0..<size {
        let vectorIndex = startIndex + i
        if let vectorPtr = reader.getVectorPointer(at: vectorIndex) {
            // Store the hash for this vector for later reference
            if let hash = reader.getHash(at: vectorIndex) {
                indexToHash[vectorIndex] = hash
            }
            
            // Copy vector data to batch array
            let batchVectorPtr = batchVectors.dataPointer.bindMemory(
                to: Float32.self, 
                capacity: Int(dimensions) * size)
            
            // Copy each dimension of the vector
            for d in 0..<Int(dimensions) {
                batchVectorPtr[i * Int(dimensions) + d] = vectorPtr.advanced(by: d).pointee
            }
        }
    }
    
    print("Batch vectors prepared, shape: \(batchVectors.shape)")
    
    // Perform prediction for this batch
    let batchPredictionStartTime = Date()
    let batchPredictionOutput = try model.prediction(input1: queryVector, input2: batchVectors)
    
    let batchCalculationTime = Date().timeIntervalSince(batchPredictionStartTime) * 1000
    print("Batch \(batchIndex + 1) similarity calculation time: \(batchCalculationTime) milliseconds")
    
    // Extract similarity scores for this batch
    guard let batchSimilarityScores = batchPredictionOutput.featureValue(for: "var_5")?.multiArrayValue else {
        print("Error: Could not extract similarity scores from batch \(batchIndex + 1)")
        return [:]
    }
    
    print("Batch \(batchIndex + 1) similarity scores shape: \(batchSimilarityScores.shape)")
    
    // Process similarity scores from this batch
    let batchScorePtr = batchSimilarityScores.dataPointer.bindMemory(to: Float32.self, capacity: size)
    var batchResults: [Int: Float] = [:]
    
    // Store scores in the dictionary with global indices
    for i in 0..<size {
        let globalIndex = startIndex + i
        let similarityScore = batchScorePtr[i]
        
        // Only store if it's not the query vector (index 0)
        if globalIndex > 0 {
            batchResults[globalIndex] = similarityScore
        }
        
        // Print scores for the first few vectors in each batch for debugging
        if i < 5 || i == size - 1 {
            print("  Vector \(globalIndex): score = \(similarityScore)")
        } else if i == 5 {
            print("  ... (more scores not shown)")
        }
    }
    
    return batchResults
}

/// Finds the top K similarity results from all batch scores
/// - Parameters:
///   - allScores: Dictionary of indices to similarity scores
///   - indexToHash: Mapping from indices to hash values
///   - topK: Number of top results to return
/// - Returns: Array of top similarity results
func findTopResults(
    from allScores: [Int: Float],
    indexToHash: [Int: UInt64],
    topK: Int = 5
) -> [VectorSimilarityResult] {
    // Find the top K scores across all batches
    let sortedResults = allScores.sorted { $0.value > $1.value }
    let topResults = sortedResults.prefix(topK)
    
    print("\nTop \(topResults.count) most similar vectors:")
    
    // Convert to our result format with index, score, and hash
    var finalResults: [VectorSimilarityResult] = []
    
    for (index, score) in topResults {
        if let hash = indexToHash[index] {
            finalResults.append((index, score, hash))
            print("  Vector \(index): \(score), Hash: \(hash)")
        }
    }
    
    return finalResults
}

// MARK: - C Exported Functions

/// Computes cosine similarity between vectors using CoreML
/// - Parameters:
///   - modelPathCStr: Path to the CoreML model file
///   - queryVectorPtr: Pointer to query vector data
///   - queryVectorDimensions: Number of dimensions in the query vector
///   - vectorsPtr: Pointer to vectors to compare with
///   - vectorCount: Number of vectors to compare with
///   - vectorDimensions: Dimensions of each vector
///   - hashesPtr: Optional pointer to hash values for the vectors
///   - resultCountPtr: Pointer to store the number of results
/// - Returns: Pointer to array of SimilarityResult structures

@_cdecl("compute_cosine_similarity")
public func compute_cosine_similarity(
    modelPathCStr: UnsafePointer<CChar>,
    queryVectorPtr: UnsafePointer<Float32>,
    queryVectorDimensions: Int32,
    vectorsPtr: UnsafePointer<Float32>,
    vectorCount: Int32,
    vectorDimensions: Int32,
    hashesPtr: UnsafePointer<UInt64>?,
    resultCountPtr: UnsafeMutablePointer<Int32>
) -> UnsafeMutablePointer<SimilarityResult>? {
    // Initialize result count to 0 by default
    resultCountPtr.pointee = 0
    
    // Convert C strings to Swift strings
    let modelPath = String(cString: modelPathCStr)
    
    print("Swift: compute_cosine_similarity called")
    print("Swift: modelPath = \(modelPath)")
    print("Swift: queryVectorDimensions = \(queryVectorDimensions), vectorCount = \(vectorCount)")
    
    do {
        // Step 1: Load the similarity model
        print("Swift: Loading cosine similarity model...")
        let model = try loadCosineSimilarityModel(from: modelPath)
        
        // Step 2: Create query vector MLMultiArray
        let queryVector = try createVectorFromPointer(pointer: queryVectorPtr, dimensions: queryVectorDimensions)
        
        // Step 3: Create vectors MLMultiArray
        let vectorsArray = try MLMultiArray(shape: [NSNumber(value: vectorCount), NSNumber(value: vectorDimensions)], 
                                         dataType: .float32)
        
        // Copy vectors data from the provided pointer
        let vectorsArrayPtr = vectorsArray.dataPointer.bindMemory(to: Float32.self, 
                                                               capacity: Int(vectorCount * vectorDimensions))
        
        for i in 0..<Int(vectorCount) {
            for j in 0..<Int(vectorDimensions) {
                let sourceIndex = i * Int(vectorDimensions) + j
                let destIndex = i * Int(vectorDimensions) + j
                vectorsArrayPtr[destIndex] = vectorsPtr.advanced(by: sourceIndex).pointee
            }
        }
        
        // Step 4: Extract hashes if provided
        var hashes: [UInt64]? = nil
        if let hashesPtr = hashesPtr {
            hashes = []
            for i in 0..<Int(vectorCount) {
                hashes!.append(hashesPtr.advanced(by: i).pointee)
            }
        }
        
        // Step 5: Compute cosine similarity
        let results = try computeCosineSimilarity(
            using: model,
            queryVector: queryVector,
            vectors: vectorsArray,
            hashes: hashes
        )
        
        // Step 6: Return the results
        if !results.isEmpty {
            // Limit to top 5 results by default
            let topResults = Array(results.prefix(5))
            resultCountPtr.pointee = Int32(topResults.count)
            return createResultArray(topResults)
        } else {
            print("No similarity results found")
            return nil
        }
    } catch {
        print("Error in compute_cosine_similarity: \(error)")
        return nil
    }
}

@_cdecl("retrieve_similar_vectors_from_corpus")
public func retrieve_similar_vectors_from_corpus(
    modelPathCStr: UnsafePointer<CChar>,
    corpusDirCStr: UnsafePointer<CChar>,
    queryVectorPtr: UnsafePointer<Float32>,
    queryVectorDimensions: Int32,
    k: Int32,
    resultCountPtr: UnsafeMutablePointer<Int32>
) -> UnsafeMutablePointer<SimilarityResult>? {
    // Initialize result count to 0 by default
    resultCountPtr.pointee = 0
    
    // Convert C strings to Swift strings
    let modelPath = String(cString: modelPathCStr)
    let corpusDir = String(cString: corpusDirCStr)
    
    print("Swift: retrieve_similar_vectors_from_corpus called")
    print("Swift: modelPath = \(modelPath)")
    print("Swift: corpusDir = \(corpusDir)")
    print("Swift: k = \(k)")
    
    do {
        // Step 1: Load the model once for reuse
        let model = try loadCosineSimilarityModel(from: modelPath)
        
        // Step 2: Create query vector MLMultiArray
        let queryVector = try createVectorFromPointer(pointer: queryVectorPtr, dimensions: queryVectorDimensions)
        
        // Step 3: List all vector dump files in the corpus directory
        let fileManager = FileManager.default
        let corpusURL = URL(fileURLWithPath: corpusDir)
        
        // Get all files with .vecdump extension
        let directoryContents = try fileManager.contentsOfDirectory(at: corpusURL, 
                                                                   includingPropertiesForKeys: nil)
        let dumpFiles = directoryContents.filter { $0.pathExtension == "vecdump" }
        
        print("Found \(dumpFiles.count) vector dump files in corpus directory")
        
        // Step 4: Process each dump file to find potential matches
        var allResults: [VectorSimilarityResult] = []
        
        for dumpFile in dumpFiles {
            // Get results from this dump file
            let dumpResults = try get_relevant_vecs_from_dump(
                modelPath: modelPath,
                vectorDumpPath: dumpFile.path,
                queryVector: queryVector,
                model: model
            )
            
            // Add to combined results
            allResults.append(contentsOf: dumpResults)
        }
        
        // Step 5: Sort all results by similarity score
        allResults.sort { $0.score > $1.score }
        
        // Step 6: Take top k results
        let topResults = Array(allResults.prefix(Int(k)))
        
        // Step 7: Return results
        if !topResults.isEmpty {
            resultCountPtr.pointee = Int32(topResults.count)
            return createResultArray(topResults)
        } else {
            print("No similarity results found in corpus")
            return nil
        }
    } catch {
        print("Error in retrieve_similar_vectors_from_corpus: \(error)")
        return nil
    }
}

/// Helper function to get relevant vectors from a dump file
/// - Parameters:
///   - modelPath: Path to the CoreML model
///   - vectorDumpPath: Path to the vector dump file
///   - queryVector: Query vector as MLMultiArray
///   - model: Optional pre-loaded model to reuse
/// - Returns: Array of similarity results
/// - Throws: If processing fails
func get_relevant_vecs_from_dump(
    modelPath: String,
    vectorDumpPath: String,
    queryVector: MLMultiArray,
    model: CosineSimilarityBatched? = nil
) throws -> [VectorSimilarityResult] {
    print("Processing dump file: \(vectorDumpPath)")
    
    // Step 1: Open the vector dump file
    let reader = VecDumpReader()
    guard reader.open(filePath: vectorDumpPath) else {
        throw NSError(domain: "NpuAPI", code: 3, 
                     userInfo: [NSLocalizedDescriptionKey: "Failed to open vector dump file"])
    }
    
    // Print vector dump info
    reader.printInfo()
    
    // Use provided model or load a new one
    let similarityModel = model ?? try loadCosineSimilarityModel(from: modelPath)
    
    // Step 2: Process vectors in batches to find similar ones
    let totalVectors = reader.count
    let dimensions = reader.dimensions
    let batchSize = min(BATCH_SIZE, totalVectors)
    var results: [VectorSimilarityResult] = []
    
    // Process in a single batch for now, can be optimized to use multiple batches if needed
    if let vectors = reader.getAllVectorsAsMLMultiArray() {
        // Prepare array of hashes
        var hashes: [UInt64] = []
        for i in 0..<totalVectors {
            if let hash = reader.getHash(at: i) {
                hashes.append(hash)
            } else {
                hashes.append(UInt64(i))
            }
        }
        
        // Compute similarity
        let batchResults = try computeCosineSimilarity(
            using: similarityModel,
            queryVector: queryVector,
            vectors: vectors,
            hashes: hashes
        )
        
        results.append(contentsOf: batchResults)
    }
    
    // Clean up resources
    reader.close()
    
    return results
}

@_cdecl("perform_similarity_check")
public func perform_similarity_check(
    modelPathCStr: UnsafePointer<CChar>, 
    vectorDumpPathCStr: UnsafePointer<CChar>,
    queryVectorPtr: UnsafePointer<Float32>? = nil,
    queryVectorDimensions: Int32 = 0,
    resultCountPtr: UnsafeMutablePointer<Int32>
) -> UnsafeMutablePointer<SimilarityResult>? {
    // Convert C strings to Swift strings
    let modelPath = String(cString: modelPathCStr)
    let vectorDumpPath = String(cString: vectorDumpPathCStr)
    
    print("Swift: perform_similarity_check called")
    print("Swift: modelPath = \(modelPath)")
    print("Swift: vectorDumpPath = \(vectorDumpPath)")
    
    // Initialize result count to 0 by default
    resultCountPtr.pointee = 0
    
    // Log if we received an external query vector
    if let queryPtr = queryVectorPtr, queryVectorDimensions > 0 {
        print("Swift: Using provided query vector with \(queryVectorDimensions) dimensions")
    } else {
        print("Swift: No query vector provided, will use first vector from dump file")
    }

    do {
        // Step 1: Initialize vector reader and load vectors
        print("Swift: Loading vectors from dump file...")
        let reader = VecDumpReader()
        guard reader.open(filePath: vectorDumpPath) else {
            print("Error: Failed to open vector dump file: \(vectorDumpPath)")
            return nil
        }
        
        // Print vector dump info
        reader.printInfo()
        
        // Step 2: Create or use provided query vector
        let queryVector: MLMultiArray
        
        if let queryPtr = queryVectorPtr, queryVectorDimensions > 0 {
            // Use the provided query vector
            queryVector = try MLMultiArray(shape: [1, NSNumber(value: queryVectorDimensions)], dataType: .float32)
            let queryDstPtr = queryVector.dataPointer.bindMemory(to: Float32.self, capacity: Int(queryVectorDimensions))
            
            // Copy data from the provided pointer
            for i in 0..<Int(queryVectorDimensions) {
                queryDstPtr[i] = queryPtr.advanced(by: i).pointee
            }
            
            print("Query vector created from provided data, shape: \(queryVector.shape)")
        } else {
            // Use the first vector from the dump file
            queryVector = try createQueryVector(from: reader)
        }
        
        // Step 3: Load the similarity model
        print("Swift: Loading cosine similarity model...")
        let model = try loadCosineSimilarityModel(from: modelPath)
        
        // Step 4: Process vectors in batches
        let startTime = Date()
        
        let totalVectors = reader.count
        let dimensions = reader.dimensions
        
        // Calculate number of batches needed
        let batchCount = (totalVectors + BATCH_SIZE - 1) / BATCH_SIZE // Ceiling division
        print("Processing \(totalVectors) vectors in \(batchCount) batches of up to \(BATCH_SIZE) vectors each")
        
        // Storage for results across all batches
        var indexToHash: [Int: UInt64] = [:]
        var allSimilarityScores: [Int: Float] = [:]
        
        // Process each batch
        for batchIndex in 0..<batchCount {
            let batchStartIndex = batchIndex * BATCH_SIZE
            let batchEndIndex = min(batchStartIndex + BATCH_SIZE, totalVectors)
            let batchSize = batchEndIndex - batchStartIndex
            
            // Process this batch and merge results
            let batchScores = try processBatch(
                index: batchIndex, 
                of: batchCount,
                startIndex: batchStartIndex,
                size: batchSize,
                dimensions: dimensions,
                reader: reader,
                model: model,
                queryVector: queryVector,
                indexToHash: &indexToHash
            )
            
            // Merge batch scores into overall results
            allSimilarityScores.merge(batchScores) { (_, new) in new }
        }
        
        let totalProcessingTime = Date().timeIntervalSince(startTime) * 1000
        print("\nTotal processing time: \(totalProcessingTime) milliseconds")
        
        // Step 5: Find top results across all batches
        let finalResults = findTopResults(from: allSimilarityScores, indexToHash: indexToHash)
        
        // Return results if we found any
        if !finalResults.isEmpty {
            print("\nReturning \(finalResults.count) top matches to C++")
            
            // Set the result count and create the array for C++
            resultCountPtr.pointee = Int32(finalResults.count)
            let resultArray = createResultArray(finalResults)
            
            // Clean up resources
            reader.close()
            return resultArray
        } else {
            print("Error: No similarity matches found")
            reader.close()
            return nil
        }
    } catch {
        print("Error during vector similarity calculation: \(error)")
        print("Localized Description: \(error.localizedDescription)")
        
        // Attempt to determine more specific error type
        if let mlError = error as? MLError {
            print("CoreML Error: \(mlError.errorDescription)")
        }
        
        return nil // Indicate failure with no results
    }
}

// MARK: - CoreML Error Handling

extension MLError {
    /// Returns a human-readable description of MLError codes
    var errorDescription: String {
        switch self.code {
        case .featureType:
            return "Invalid feature type"
        case .genericError:
            return "Generic CoreML error"
        case .invalidImageSize:
            return "Invalid image size for model input"
        case .labelsNotFound:
            return "Labels not found"
        case .modelCreation:
            return "Model creation failed"
        case .modelMismatch:
            return "Model mismatch"
        case .notProcessed:
            return "Not processed"
        case .parameterError:
            return "Parameter error"
        @unknown default:
            return "Unknown CoreML error code: \(self.code.rawValue)"
        }
    }
}
