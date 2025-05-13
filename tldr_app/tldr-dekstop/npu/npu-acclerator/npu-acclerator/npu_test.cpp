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

    // Define the path to the compiled Core ML model and corpus directory
    const char* modelPath = "/Users/manu/proj_tldr/tldr-dekstop/release-products/artefacts/CosineSimilarityBatched.mlmodelc";
    const char* corpusDir = "/Users/manu/proj_tldr/corpus/current/"; // Example directory, adjust as needed
    std::cout << "C++: Using model path: " << modelPath << std::endl;
    std::cout << "C++: Using corpus directory: " << corpusDir << std::endl;

    // Example query vector (replace with real data as needed)
    const int32_t queryVectorDimensions = 384;
    float queryVector[queryVectorDimensions] = {0};
    for (int i = 0; i < queryVectorDimensions; ++i) queryVector[i] = static_cast<float>(i) / queryVectorDimensions;

    // Number of top results to retrieve
    int32_t k = 5;
    int32_t resultCount = 0;

    // Call the Swift function
    SimilarityResult* results = retrieve_similar_vectors_from_corpus(
        modelPath,
        corpusDir,
        queryVector,
        queryVectorDimensions,
        k,
        &resultCount
    );

    if (results && resultCount > 0) {
        std::cout << "Top " << resultCount << " similar vectors:" << std::endl;
        for (int i = 0; i < resultCount; ++i) {
            std::cout << "Hash: " << results[i].hash << ", Score: " << results[i].score << std::endl;
        }
    } else {
        std::cout << "No results or function failed." << std::endl;
    }

    // Free the results memory if allocated
    if (results) free_similarity_results(results);

    return 0;
}

