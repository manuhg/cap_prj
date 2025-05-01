//
//  trial.cpp
//  npu-acclerator
//
//  Created by Manu Hegde on 5/1/25.
//


#include <iostream>
#include "npu_accelerator.h" // Include the C header for the Swift function

int main() {
    std::cout << "Calling Swift function from C++..." << std::endl;

    // Define the path to the compiled Core ML model relative to the executable
    const char* modelPath = "CosineSimilarityBatched.mlmodelc";
    std::cout << "C++: Using model path: " << modelPath << std::endl;

    // Call the Swift function and pass the path
    int result = perform_similarity_check(modelPath);

    if (result == 0) {
        std::cout << "Swift function executed successfully." << std::endl;
    } else {
        std::cout << "Swift function failed with code: " << result << std::endl;
    }

    return 0;
}
