#include <iostream>
#include "tldr.h"

int main() {
    // Initialize system
    if (!tldr::initializeSystem()) {
        std::cerr << "Failed to initialize system" << std::endl;
        return 1;
    }

    std::string testFile = "~/proj_tldr/corpus/current/0.System Design Interview An Insider’s Guide by Alex Xu.pdf";

    tldr::addCorpus(testFile);
    tldr::queryRag("What does the book say about hotspot problem?");

    // Cleanup system
    tldr::cleanupSystem();

    return 0;
}
