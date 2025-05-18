#include <iostream>
#include "lib_tldr/tldr_api.h"

int main() {
    // Initialize system
    if (!tldr_cpp_api::initializeSystem()) {
        std::cerr << "Failed to initialize system" << std::endl;
        return 1;
    }

    // std::string testFile = "~/proj_tldr/corpus/current/0.System Design Interview An Insider’s Guide by Alex Xu.pdf";

    // tldr_cpp_api::addCorpus(testFile);
    tldr_cpp_api::addCorpus("~/proj_tldr/corpus/current");
    // tldr_cpp_api::queryRag("What does is hotspot problem in cache?");

    // Cleanup system
    tldr_cpp_api::cleanupSystem();

    return 0;
}
