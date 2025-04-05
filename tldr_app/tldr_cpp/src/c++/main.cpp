#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <pqxx/pqxx>
#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-page.h>

#include <main.h>
#include <regex>
#include <curl/curl.h>

////tests
using json = nlohmann::json;

void test_extractTextFromPDF() {
    std::string testFile = "../../corpus/current/97-things-every-software-architect-should-know.pdf";

    std::string extractedText = extractTextFromPDF(testFile);
    std::cout << extractedText.length() << std::endl;

    assert(extractedText.length()==268979);
    std::cout << "test_extractTextFromPDF passed!" << std::endl;
    std::cout << "Extracted text (first 100 chars): \n=========" << extractedText.substr(0, 100) << "\n=========" <<
            std::endl;
}

//////
// Simple function to extract text from a PDF file
std::string extractTextFromPDF(const std::string &filename) {
    // Load the PDF document
    poppler::document *doc = poppler::document::load_from_file(filename);

    // Check if the document loaded successfully
    if (!doc) {
        std::cerr << "Error opening PDF file." << std::endl;
        return "";
    }

    std::string text;

    // Iterate through all pages
    for (int i = 0; i < doc->pages(); ++i) {
        // Get the current page
        poppler::page *page = doc->create_page(i);
        if (page) {
            // Extract text from the page and append it to our result
            poppler::byte_array utf8_data = page->text().to_utf8();
            std::string page_text(utf8_data.begin(), utf8_data.end());

            text.append(page_text + PAGE_DELIMITER);
            delete page;
        }
    }

    // Clean up
    delete doc;

    return text;
}

// function to split text paragraphs into chunks with 20 characters overlap at the start and end
std::vector<std::string> splitTextIntoChunks(const std::string &text, size_t num_sentences, size_t overlap) {
    std::vector<std::string> chunks;
    size_t start = 0;
    // do regex search to find sentence endings
    std::regex sentenceEndRegex(R"([\.\!\?]+)");
    // find locations of next 10 sentence endings
    std::smatch match;
    std::string::const_iterator searchStart(text.cbegin());
    std::vector<size_t> sentences;
    while (std::regex_search(searchStart, text.cend(), match, sentenceEndRegex)) {
        sentences.push_back(match.position() + match.length());
        searchStart = match.suffix().first;
    }
    // use sentence locations to split text into chunks based on num_sentences
    for (size_t i = 0; i < sentences.size(); i += num_sentences) {
        size_t end = (i + num_sentences < sentences.size()) ? sentences[i + num_sentences] : text.length();
        std::string chunk = text.substr(start, end - start);
        if (i > 0) {
            chunk = text.substr(sentences[i - 1] - overlap, end - start + overlap);
        }
        chunks.push_back(chunk);
        start = end;
    }

    return chunks;
}


void sendEmbeddingsRequest(const std::vector<std::string> &chunks) {
    CURL *curl;
    CURLcode res;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        std::string url = LLM_EMBEDDINGS_URL;
        // convert chunks into json data
        embeddings_request request =  {chunks};
        json request_json = request;


        std::string jsonData = request_json.dump(2);
        std::cout <<"\njsonData:\n"<< jsonData << std::endl;

        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());

        //parse response
        std::string response;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            // Parse the response
            json response_json = json::parse(response);
            std::cout << "Response: " << response_json.dump(2) << std::endl;
        }
        // Clean up
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    curl_global_cleanup();
}

void doRag(const std::string &conversationId) {
    // Implement the function
    std::cout << "DO_RAG action with conversation_id: " << conversationId << std::endl;

    try {
        pqxx::connection C("dbname=testdb user=postgres password=secret hostaddr=127.0.0.1 port=5432");
        pqxx::work W(C);

        std::string query =
                "SELECT * FROM conversations WHERE id = " + W.quote(conversationId) + " ORDER BY created_at";
        pqxx::result R = W.exec(query);

        for (auto row: R) {
            std::cout << "ID: " << row["id"].c_str() << ", Created At: " << row["created_at"].c_str() << std::endl;
        }

        W.commit();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}

void addCorpus(const std::string &sourcePath) {
    std::string doc_text = extractTextFromPDF(sourcePath);
    if (doc_text.empty()) {
        std::cerr << "Error: No text extracted from PDF." << std::endl;
        return;
    }
    // print text length
    std::cout << "Extracted text length: " << doc_text.length() << std::endl;
    const std::vector<std::string> chunks = splitTextIntoChunks(doc_text, CHUNK_N_SENTENCES, CHUNK_N_OVERLAP);
    std::cout << "Number of chunks: " << chunks.size() << std::endl;
    sendEmbeddingsRequest(chunks);
    //TODO save the json to database and return the id
}

void deleteCorpus(const std::string &corpusId) {
    // Implement the function
    std::cout << "DELETE_CORPUS action with corpus_id: " << corpusId << std::endl;
}

void command_loop() {
    std::string input;
    std::map<std::string, std::function<void(const std::string &)> > actions = {
        {"do-rag", doRag},
        {"add-corpus", addCorpus},
        {"delete-corpus", deleteCorpus}
    };

    while (true) {
        std::cout << "Enter command: ";
        std::getline(std::cin, input);
        std::istringstream iss(input);
        std::string command, argument;
        iss >> command >> argument;

        auto it = actions.find(command);
        if (it != actions.end()) {
            it->second(argument);
        } else if (command == "exit") {
            break;
        } else {
            std::cout << "Unknown command: " << command << std::endl;
        }
    }
}


int main() {
    test_extractTextFromPDF();
    // command_loop();
    return 0;
}

