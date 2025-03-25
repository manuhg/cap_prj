//
// Created by Manu Hegde on 3/25/25.
//

#include "test_pdf.h"

#include <cassert>
#include "../src/c++/main.cpp"
void test_extractTextFromPDF() {
    std::string testFile = "../corpus/current/97-things-every-software-architect-should-know.pdf";
    std::string expectedText = "Expected text from the PDF file";

    std::string extractedText = extractTextFromPDF(testFile);

    assert(extractedText == expectedText);
    std::cout << "test_extractTextFromPDF passed!" << std::endl;
}