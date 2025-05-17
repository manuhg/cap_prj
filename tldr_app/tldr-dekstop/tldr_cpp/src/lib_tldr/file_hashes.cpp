#include "lib_tldr.h"
#include <array>
#include <memory>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <cctype>

// Helper function to execute a shell command and get its output
static std::string exec(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::map<std::string, std::string> computeFileHashes(const std::vector<std::string>& file_paths) {
    std::map<std::string, std::string> file_hashes;
    
    // Check if shasum is available
    if (system("which shasum > /dev/null 2>&1") != 0) {
        throw std::runtime_error("shasum command not found. Please install shasum utility.");
    }
    
    // Process files in chunks to avoid command line length limits
    const size_t chunk_size = 50;
    for (size_t i = 0; i < file_paths.size(); i += chunk_size) {
        auto start = file_paths.begin() + i;
        auto end = (i + chunk_size <= file_paths.size()) ? start + chunk_size : file_paths.end();
        
        // Build the command
        std::string cmd = "shasum -a 256 ";
        for (auto it = start; it != end; ++it) {
            // Escape special characters in file paths
            std::string escaped_path = "\"" + *it + "\" ";
            cmd += escaped_path;
        }
        
        // Execute the command and get the output
        std::string output;
        try {
            output = exec(cmd);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to execute shasum command: ") + e.what());
        }
        
        // Parse the output
        std::istringstream iss(output);
        std::string line;
        size_t current_idx = i;
        
        while (std::getline(iss, line) && current_idx < file_paths.size()) {
            // Split the line into hash and filename
            size_t space_pos = line.find(' ');
            if (space_pos == std::string::npos) continue;
            
            std::string hash = line.substr(0, 64); // SHA-256 is 64 chars
            std::string file_path = file_paths[current_idx];
            
            // Verify the hash is valid (64 hex chars)
            if (hash.length() == 64 && 
                std::all_of(hash.begin(), hash.end(), [](char c) {
                    return std::isxdigit(static_cast<unsigned char>(c));
                })) {
                file_hashes[file_path] = hash;
            } else {
                throw std::runtime_error("Invalid hash format for file: " + file_path);
            }
            
            current_idx++;
        }
    }
    
    return file_hashes;
}
