import CoreML
import Foundation


// Define the input type for the model
class CosineSimilarityInputCustom: MLFeatureProvider {
    var input1: MLMultiArray
    var input2: MLMultiArray

    var featureNames: Set<String> {
        return ["input1", "input2"]
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "input1":
            return MLFeatureValue(multiArray: input1)
        case "input2":
            return MLFeatureValue(multiArray: input2)
        default:
            return nil
        }
    }

    init(input1: MLMultiArray, input2: MLMultiArray) {
        self.input1 = input1
        self.input2 = input2
    }
}

class CosineSimilarityModel {
    private let model: MLModel

    init() throws {
        // Load the CoreML model
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine //.all // Use all available compute units, including NPU
        self.model = try MLModel(contentsOf: URL(fileURLWithPath: "/Users/manu/dev/UW/cap_prj/tldr_app/tldr-dekstop/npu/npu-acclerator/npu-acclerator/CosineSimilarity.mlpackage"), configuration: config)
    }

    func predict(input1: MLMultiArray, input2: MLMultiArray) throws -> Double {
        // Prepare inputs
        let inputs = CosineSimilarityInputCustom(input1: input1, input2: input2)
        
        // Make prediction
        let prediction = try model.prediction(from: inputs)
        
        // Extract the similarity score
        guard let similarity = prediction.featureValue(for: "var_5")?.doubleValue else {
            throw NSError(domain: "CosineSimilarityModel", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to extract similarity"])
        }
        return similarity
    }
}

// Function to generate random MLMultiArray
func randomVector(size: Int) -> MLMultiArray {
    let array = try! MLMultiArray(shape: [NSNumber(value: size)], dataType: .double)
    for i in 0..<size {
        array[i] = NSNumber(value: Double.random(in: 0...1))
    }
    return array
}

// Main function to compare a vector with 100 other vectors
func main() {
    let vectorSize = 10
    let baseVector = randomVector(size: vectorSize)
    let model = try! CosineSimilarityModel()
    
    for i in 1...100 {
        let comparisonVector = randomVector(size: vectorSize)
        let similarity = try! model.predict(input1: baseVector, input2: comparisonVector)
        print("Similarity with vector \(i): \(similarity)")
    }
}

main()
