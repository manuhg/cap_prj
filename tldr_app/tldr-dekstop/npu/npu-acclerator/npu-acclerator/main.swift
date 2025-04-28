import Foundation
import CoreML

// Constants
let VECTOR_DIM = 384
let BATCH_SIZE = 1024*128 // Example batch size (must be <= 1024)

// Helper function to create MLMultiArray with specific shape and fill value
func createVector(shape: [NSNumber], value: Float) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape, dataType: .float32)
    let count = array.count
    let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: count)
    for i in 0..<count {
        pointer[i] = value
    }
    return array
}

// Helper function to create MLMultiArray with random values
func createRandomVector(shape: [NSNumber]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape, dataType: .float32)
    let count = array.count
    let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: count)
    for i in 0..<count {
        pointer[i] = Float.random(in: -1.0...1.0) // Example random range
    }
    return array
}

// Main function to test batched cosine similarity
func main() {
    do {
        // Ensure the new model is targeted in Xcode and the old one removed/untargeted
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine // use cpu and neural engine but not the GPU

        let model = try CosineSimilarityBatched(configuration: configuration)
        let startTime0 = Date()

        print("Testing Batched Cosine Similarity (batch size: \(BATCH_SIZE)) with random vectors")

        // Create base vector (input1: shape [1, VECTOR_DIM]) - Random
        let baseVector = try createRandomVector(shape: [1, NSNumber(value: VECTOR_DIM)])
        print("Base Vector (input1) shape: \(baseVector.shape)")

        // Create batch of comparison vectors (input2: shape [BATCH_SIZE, VECTOR_DIM]) - Random
        let comparisonVectors = try createRandomVector(shape: [NSNumber(value: BATCH_SIZE), NSNumber(value: VECTOR_DIM)])
        print("Comparison Vectors (input2) shape: \(comparisonVectors.shape)")
        let elapsedTime0 = Date().timeIntervalSince(startTime0) * 1000
           print("Data prep time: \(elapsedTime0) milliseconds\n\n")
        
        let startTime = Date()

        // Perform prediction
        let predictionOutput = try model.prediction(input1: baseVector, input2: comparisonVectors)
        
        let elapsedTime = Date().timeIntervalSince(startTime) * 1000
           print("Execution time: \(elapsedTime) milliseconds")

        // Extract the result (output name is 'var_5' based on latest conversion log)
        guard let similarityScores = predictionOutput.featureValue(for: "var_5")?.multiArrayValue else {
            print("Error: Could not extract similarity scores multiArrayValue from output.")
            return
        }

        print("Output Similarities (var_5) shape: \(similarityScores.shape)")

        // Process the output similarity scores (shape [BATCH_SIZE])
        let count = similarityScores.count
        if count == BATCH_SIZE {
//            let pointer = similarityScores.dataPointer.bindMemory(to: Float32.self, capacity: count)
            print("Calculated Similarities:")
//            for i in 0..<count {
//                print("  Similarity with vector \(i): \(pointer[i])")
//            }
        } else {
            print("Error: Output similarity count (\(count)) does not match BATCH_SIZE (\(BATCH_SIZE))")
        }

    } catch {
        print("Error during model loading or prediction: \(error)")
    }
}

main()
