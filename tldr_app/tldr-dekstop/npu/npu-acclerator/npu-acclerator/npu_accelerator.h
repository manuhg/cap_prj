#ifndef NPU_ACCELERATOR_H
#define NPU_ACCELERATOR_H

#ifdef __cplusplus
extern "C" {
#endif

// Declare the Swift function callable from C/C++
// Returns 0 on success, -1 on failure.
int perform_similarity_check(const char* modelPath);

#ifdef __cplusplus
}
#endif

#endif // NPU_ACCELERATOR_H
