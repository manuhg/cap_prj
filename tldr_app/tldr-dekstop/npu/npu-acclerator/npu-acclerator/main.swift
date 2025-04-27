import CoreML
import Foundation


// Define the input type for the model
class CosineSimilarityInput: MLFeatureProvider {
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
        self.model = try MLModel(contentsOf: URL(fileURLWithPath: "CosineSimilarity.mlpackage"), configuration: config)
    }

    func predict(input1: MLMultiArray, input2: MLMultiArray) throws -> Double {
        // Prepare inputs
        let inputs = CosineSimilarityInput(input1: input1, input2: input2)
        
        // Make prediction
        let prediction = try model.prediction(from: inputs)
        
        // Extract the similarity score
        guard let similarity = prediction.featureValue(for: "var_5")?.doubleValue else {
            throw NSError(domain: "CosineSimilarityModel", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to extract similarity"])
        }
        return similarity
    }
}
