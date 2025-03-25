//
//

#ifndef TLDR_CPP_MAIN_H
#define TLDR_CPP_MAIN_H

// tests
void test_extractTextFromPDF();
//
std::string extractTextFromPDF(const std::string &filename);
void doRag(const std::string &conversationId);
void addCorpus(const std::string &sourcePath);
void deleteCorpus(const std::string &corpusId);
void command_loop();

#endif //TLDR_CPP_MAIN_H
