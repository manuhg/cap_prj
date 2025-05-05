#pragma once

#include <vector>
#include <string>
#include <memory>
#include <sys/mman.h>
#include <unistd.h>

namespace tldr {

// Structure for the header of the vector cache dump file
struct VectorCacheDumpHeader {
    uint32_t num_entries;        // Number of embedding vectors/hashes
    uint32_t hash_size_bytes;    // Size of each hash in bytes
    uint32_t vector_size_bytes;  // Size of each embedding vector in bytes
    uint32_t vector_dimensions;  // Number of dimensions in each vector
};

// Structure to hold memory-mapped vector data
struct MappedVectorData {
    void* mapped_memory = nullptr;         // Raw memory mapping pointer
    size_t file_size = 0;                  // Size of the mapped file
    int fd = -1;                           // File descriptor
    VectorCacheDumpHeader* header = nullptr; // Pointer to the header
    const float* vectors = nullptr;        // Pointer to the vectors array
    const size_t* hashes = nullptr;        // Pointer to the hashes array
    
    // Cleanup resources
    ~MappedVectorData() {
        if (mapped_memory && mapped_memory != MAP_FAILED) {
            munmap(mapped_memory, file_size);
        }
        if (fd != -1) {
            close(fd);
        }
    }
};

// Dump vectors and hashes to a binary file for memory mapping
bool dump_vectors_to_file(const std::string& source_path, 
                         const std::vector<std::vector<float>>& embeddings,
                         const std::vector<size_t>& hashes);

// Read a vector dump file using memory mapping and return pointers to the data
std::unique_ptr<MappedVectorData> read_vector_dump_file(const std::string& dump_file_path);

// Print information about a mapped vector file
void print_vector_dump_info(const MappedVectorData* data, const std::string& file_path, bool print_sample = true);

// Test vector cache dump and read functionality
bool test_vector_cache();

} // namespace tldr
