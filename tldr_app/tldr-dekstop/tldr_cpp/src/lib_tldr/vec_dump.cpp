#include "vec_dump.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sys/stat.h>
#include <fcntl.h>
#include "constants.h"

namespace tldr {

// Dump vectors and hashes to a binary file for memory mapping
bool dump_vectors_to_file(const std::string& source_path, 
                         const std::vector<std::vector<float>>& embeddings,
                         const std::vector<uint64_t>& hashes,
                         const std::string& fileHash) {
    
    if (embeddings.empty() || hashes.empty() || embeddings.size() != hashes.size()) {
        std::cerr << "Error: Invalid embeddings or hashes for dumping to file" << std::endl;
        return false;
    }

    // Create a directory for vecdump files if it doesn't exist
    if (!std::filesystem::exists(VECDUMP_DIR)) {
        std::filesystem::create_directory(VECDUMP_DIR);
    }

    // Create vecdump filename using the file hash
    std::string vecdump_path = std::string(VECDUMP_DIR) + "/" + fileHash + ".vecdump";
    
    // Log the save operation
    std::cout << "Vecdump saved to: " << vecdump_path << std::endl;
    
    // Extract the base filename from source path
    std::filesystem::path path(source_path);
    std::string filename = path.string() + ".vecdump";
    
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to save vecdump for " << source_path << std::endl;
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    // Prepare the header
    VectorCacheDumpHeader header;
    header.num_entries = static_cast<uint32_t>(embeddings.size());
    header.hash_size_bytes = sizeof(uint64_t);
    header.vector_dimensions = static_cast<uint32_t>(embeddings[0].size());
    header.vector_size_bytes = sizeof(float) * header.vector_dimensions;
    
    // Write the header
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    // Write all embedding vectors as a continuous block
    for (const auto& embedding : embeddings) {
        if (embedding.size() != header.vector_dimensions) {
            std::cerr << "Error: Inconsistent embedding vector dimensions" << std::endl;
            return false;
        }
        
        out.write(reinterpret_cast<const char*>(embedding.data()), 
                 header.vector_size_bytes);
    }
    
    // Write all hashes as a continuous block
    out.write(reinterpret_cast<const char*>(hashes.data()), 
              header.hash_size_bytes * header.num_entries);
    
    out.close();
    
    std::cout << "Successfully wrote vector cache to " << filename << std::endl;
    std::cout << "  Entries: " << header.num_entries 
              << ", Vector dim: " << header.vector_dimensions
              << ", Total size: " << (sizeof(header) + 
                                    header.num_entries * header.vector_size_bytes + 
                                    header.num_entries * header.hash_size_bytes) 
              << " bytes" << std::endl;
    
    return true;
}

// Read a vector dump file using memory mapping and return pointers to the data
std::unique_ptr<MappedVectorData> read_vector_dump_file(const std::string& dump_file_path) {
    auto result = std::make_unique<MappedVectorData>();
    
    // Open the file
    result->fd = open(dump_file_path.c_str(), O_RDONLY);
    if (result->fd == -1) {
        std::cerr << "Error opening file: " << dump_file_path << std::endl;
        return nullptr;
    }
    
    // Get file size
    struct stat sb;
    if (fstat(result->fd, &sb) == -1) {
        std::cerr << "Error getting file stats" << std::endl;
        return nullptr;
    }
    result->file_size = sb.st_size;
    
    // Memory map the file
    result->mapped_memory = mmap(NULL, result->file_size, PROT_READ, MAP_PRIVATE, result->fd, 0);
    if (result->mapped_memory == MAP_FAILED) {
        std::cerr << "Error memory mapping the file" << std::endl;
        return nullptr;
    }
    
    // Set up the header pointer
    result->header = static_cast<VectorCacheDumpHeader*>(result->mapped_memory);
    
    // Calculate offsets
    size_t header_size = sizeof(VectorCacheDumpHeader);
    size_t vectors_section_size = result->header->num_entries * result->header->vector_size_bytes;
    
    // Set up pointers to the vectors and hashes sections
    result->vectors = reinterpret_cast<const float*>(
        static_cast<const char*>(result->mapped_memory) + header_size);
        
    result->hashes = reinterpret_cast<const uint64_t*>(
        static_cast<const char*>(result->mapped_memory) + header_size + vectors_section_size);
    
    return result;
}

// Print information about a mapped vector file
void print_vector_dump_info(const MappedVectorData* data, const std::string& file_path, bool print_sample) {
    if (!data || !data->header) {
        std::cerr << "Error: Invalid mapped vector data" << std::endl;
        return;
    }
    
    std::cout << "=== Vector Cache File: " << file_path << " ===" << std::endl;
    std::cout << "Number of entries: " << data->header->num_entries << std::endl;
    std::cout << "Hash size (bytes): " << data->header->hash_size_bytes << std::endl;
    std::cout << "Vector size (bytes): " << data->header->vector_size_bytes << std::endl;
    std::cout << "Vector dimensions: " << data->header->vector_dimensions << std::endl;
    
    // Print sample entry (index 1) if available and requested
    if (print_sample && data->header->num_entries > 1) {
        size_t index = 1;  // Show element at index 1
        
        // Print hash for element at index 1
        std::cout << "\nSample element (index " << index << "):" << std::endl;
        std::cout << "Hash: " << data->hashes[index] << std::endl;
        
        // Print embedding vector for element at index 1
        std::cout << "Embedding vector (first 10 dimensions):" << std::endl;
        const float* vector = data->vectors + (index * data->header->vector_dimensions);
        
        size_t dims_to_show = std::min(static_cast<size_t>(10), 
                                    static_cast<size_t>(data->header->vector_dimensions));
        for (size_t i = 0; i < dims_to_show; i++) {
            std::cout << vector[i];
            if (i < dims_to_show - 1) std::cout << ", ";
        }
        std::cout << (data->header->vector_dimensions > 10 ? "..." : "") << std::endl;
    } else if (print_sample) {
        std::cout << "Not enough entries to show sample at index 1" << std::endl;
    }
}

// Test vector cache dump and read functionality
bool test_vector_cache() {
    std::cout << "=== Testing Vector Cache Dump and Read Functionality ===" << std::endl;
    
    // Create sample embeddings and hashes
    std::vector<std::vector<float>> test_embeddings;
    std::vector<uint64_t> test_hashes;
    
    // Create 5 test embeddings with 16 dimensions each
    const size_t num_embeddings = 5;
    const size_t dimensions = 16;
    
    std::cout << "Creating " << num_embeddings << " test embeddings with " 
              << dimensions << " dimensions each" << std::endl;
              
    // Initialize with deterministic values for testing
    for (size_t i = 0; i < num_embeddings; i++) {
        std::vector<float> embedding(dimensions);
        for (size_t j = 0; j < dimensions; j++) {
            embedding[j] = static_cast<float>((i + 1) * 0.1f + j * 0.01f); // Deterministic pattern
        }
        test_embeddings.push_back(std::move(embedding));
        
        // Create corresponding hash
        test_hashes.push_back(1000000 + i * 10000); // Simple deterministic hash for testing
    }
    
    // Test file path
    std::string test_file = "vector_cache_test.bin";
    
    // Step 1: Dump the test data
    std::cout << "\nStep 1: Dumping test embeddings to " << test_file << std::endl;
    if (!dump_vectors_to_file(test_file, test_embeddings, test_hashes, "test_hash")) {
        std::cerr << "Error: Failed to dump test embeddings" << std::endl;
        return false;
    }
    
    // Step 2: Read the dumped file
    std::cout << "\nStep 2: Reading the vector dump file" << std::endl;
    auto mapped_data = read_vector_dump_file(test_file);
    if (!mapped_data) {
        std::cerr << "Error: Failed to read the vector dump file" << std::endl;
        return false;
    }
    
    // Step 3: Print info about the file
    print_vector_dump_info(mapped_data.get(), test_file, false);
    
    // Step 4: Verify contents
    std::cout << "\nStep 4: Verifying file contents" << std::endl;
    
    // Verify header
    bool header_verified = 
        (mapped_data->header->num_entries == num_embeddings) &&
        (mapped_data->header->hash_size_bytes == sizeof(uint64_t)) &&
        (mapped_data->header->vector_dimensions == dimensions);
    
    std::cout << "Header verification: " << (header_verified ? "PASSED" : "FAILED") << std::endl;
    
    // Verify contents by checking values at index 1 (second element)
    bool data_verified = true;
    size_t test_idx = 1;
    
    if (test_idx < mapped_data->header->num_entries) {
        std::cout << "\nVerifying element at index " << test_idx << ":" << std::endl;
        
        // Original hash and the one read from file
        uint64_t original_hash = test_hashes[test_idx];
        uint64_t read_hash = mapped_data->hashes[test_idx];
        
        std::cout << "Hash verification: Original = " << original_hash 
                  << ", Read = " << read_hash 
                  << " -> " << (original_hash == read_hash ? "MATCH" : "MISMATCH") << std::endl;
        
        // Check a few dimensions of the embedding vector
        const float* read_vector = mapped_data->vectors + (test_idx * mapped_data->header->vector_dimensions);
        
        std::cout << "Vector verification (first 5 dimensions):" << std::endl;
        bool vector_matches = true;
        for (size_t i = 0; i < 5 && i < mapped_data->header->vector_dimensions; i++) {
            float original_val = test_embeddings[test_idx][i];
            float read_val = read_vector[i];
            bool matches = (std::abs(original_val - read_val) < 0.000001f); // Floating point comparison with epsilon
            
            std::cout << "  Dim " << i << ": Original = " << original_val 
                      << ", Read = " << read_val 
                      << " -> " << (matches ? "MATCH" : "MISMATCH") << std::endl;
            
            if (!matches) vector_matches = false;
        }
        
        data_verified = (original_hash == read_hash) && vector_matches;
    }
    
    // Print final result
    std::cout << "\nTest result: " << (header_verified && data_verified ? "PASSED" : "FAILED") << std::endl;
    
    return header_verified && data_verified;
}

} // namespace tldr
