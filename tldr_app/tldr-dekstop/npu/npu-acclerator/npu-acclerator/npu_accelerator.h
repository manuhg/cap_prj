#ifndef NPU_ACCELERATOR_H
#define NPU_ACCELERATOR_H

#include <stdint.h> // For uint64_t

#ifdef __cplusplus
extern "C" {
#endif

// Structure for similarity search results
// Must match Swift definition
typedef struct {
    uint64_t hash;  // Hash value of the vector
    float score;    // Similarity score (higher is more similar)
} SimilarityResult;

/**
 * Perform vector similarity search on a single dump file
 * 
 * @param modelPath Path to the CoreML model (.mlmodelc directory)
 * @param vectorDumpPath Path to the vector dump file
 * @param queryVectorPtr Pointer to query vector data (if NULL, first vector from dump is used)
 * @param queryVectorDimensions Dimensions of the query vector (ignored if queryVectorPtr is NULL)
 * @param resultCountPtr Pointer to store the number of results
 * 
 * @return Pointer to array of SimilarityResult structures (must be freed by caller)
 */
/**
 * Perform vector similarity search on a single dump file
 * @param modelPath Path to the CoreML model (.mlmodelc directory)
 * @param vectorDumpPath Path to the vector dump file
 * @param queryVectorPtr Pointer to query vector data (if NULL, first vector from dump is used)
 * @param queryVectorDimensions Dimensions of the query vector (ignored if queryVectorPtr is NULL)
 * @param resultCountPtr Pointer to store the number of results
 * @return Pointer to array of SimilarityResult structures (must be freed by caller using free_similarity_results)
 */
SimilarityResult* perform_similarity_check(
    const char* modelPath,
    const char* vectorDumpPath,
    const float* queryVectorPtr,
    int32_t queryVectorDimensions,
    int32_t* resultCountPtr
);


/**
 * Compute cosine similarity between vectors
 * 
 * @param modelPath Path to the CoreML model (.mlmodelc directory)
 * @param queryVectorPtr Pointer to query vector data
 * @param queryVectorDimensions Dimensions of the query vector
 * @param vectorsPtr Pointer to vectors to compare with
 * @param vectorCount Number of vectors to compare with
 * @param vectorDimensions Dimensions of each vector
 * @param resultCountPtr Pointer to store the number of results
 * 
 * @return Pointer to array of indices and scores (must be freed by caller)
 */
/**
 * Compute cosine similarity between vectors using CoreML model
 * @param modelPath Path to the CoreML model (.mlmodelc directory)
 * @param queryVectorPtr Pointer to query vector data
 * @param queryVectorDimensions Dimensions of the query vector
 * @param vectorsPtr Pointer to vectors to compare with
 * @param vectorCount Number of vectors to compare with
 * @param vectorDimensions Dimensions of each vector
 * @param hashesPtr Optional pointer to hashes (can be NULL)
 * @param resultCountPtr Pointer to store the number of results
 * @return Pointer to array of SimilarityResult (must be freed by caller using free_similarity_results)
 */
SimilarityResult* compute_cosine_similarity(
    const char* modelPath,
    const float* queryVectorPtr,
    int32_t queryVectorDimensions,
    const float* vectorsPtr,
    int32_t vectorCount,
    int32_t vectorDimensions,
    const uint64_t* hashesPtr,
    int32_t* resultCountPtr
);


/**
 * Find relevant vectors from a corpus directory
 * 
 * @param modelPath Path to the CoreML model (.mlmodelc directory)
 * @param corpusDir Directory containing vector dump files
 * @param queryVectorPtr Pointer to query vector data
 * @param queryVectorDimensions Dimensions of the query vector
 * @param k Number of top results to return
 * @param resultCountPtr Pointer to store the number of results
 * 
 * @return Pointer to array of SimilarityResult structures (must be freed by caller)
 */
/**
 * Find relevant vectors from a corpus directory
 * @param modelPath Path to the CoreML model (.mlmodelc directory)
 * @param corpusDir Directory containing vector dump files
 * @param queryVectorPtr Pointer to query vector data
 * @param queryVectorDimensions Dimensions of the query vector
 * @param k Number of top results to return
 * @param resultCountPtr Pointer to store the number of results
 * @return Pointer to array of SimilarityResult (must be freed by caller using free_similarity_results)
 */
SimilarityResult* retrieve_similar_vectors_from_corpus(
    const char* modelPath,
    const char* corpusDir,
    const float* queryVectorPtr,
    int32_t queryVectorDimensions,
    int32_t k,
    int32_t* resultCountPtr
);


/// Frees similarity results allocated by Swift. Must be called from C/C++ after results are no longer needed.
/**
 * Free memory allocated for similarity results by Swift.
 * @param ptr Pointer returned by similarity search functions.
 */
void free_similarity_results(void* ptr);


#ifdef __cplusplus
}
#endif

#endif // NPU_ACCELERATOR_H
