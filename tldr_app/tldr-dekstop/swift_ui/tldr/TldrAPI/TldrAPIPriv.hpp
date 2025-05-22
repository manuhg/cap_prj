//
//  TldrAPIPriv.hpp
//  TldrAPI
//
//  Created by Manu Hegde on 4/18/25.
//

#ifndef TLDRAPIPRIV_HPP
#define TLDRAPIPRIV_HPP

#include "tldr_api.h"
#include <string>
#include <vector>

// Forward declarations of C++ types
namespace tldr_cpp_api {
    struct ContextChunk {
        std::string text;
        float similarity;
        uint64_t hash;
    };

    struct RagResult {
        std::string response;
        std::vector<ContextChunk> context_chunks;
    };
}

#endif /* TLDRAPIPRIV_HPP */
