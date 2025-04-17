import os
import langchain

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)


class RAGApplication:
    def __init__(self, openai_api_key):
        """Initialize the RAG application with necessary components."""
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")

        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            show_progress_bar=False,  # Disable progress bar
        )
        self.vector_store = None
        self.qa_chain = None

        # Initialize text splitter with default parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

    def load_pdfs(self, pdf_directory):
        """Load PDFs from a directory and split them into chunks."""
        if not os.path.exists(pdf_directory):
            raise ValueError(f"Directory not found: {pdf_directory}")

        documents = []

        # Iterate through all PDF files in the directory
        for filename in os.listdir(pdf_directory):
            if filename.endswith(".pdf"):
                file_path = os.path.join(pdf_directory, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())

        if not documents:
            raise ValueError(f"No PDF files found in {pdf_directory}")

        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        return texts

    def create_vector_store(self, texts, persist_directory="./vector_store"):
        """Create and persist a vector store from the document chunks."""
        # Ensure the persist directory exists
        os.makedirs(persist_directory, exist_ok=True)

        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=persist_directory,
        )
        return self.vector_store

    def load_existing_vector_store(self, persist_directory="./vector_store"):
        """Load an existing vector store from disk."""
        if not os.path.exists(persist_directory):
            raise ValueError(f"Vector store directory not found: {persist_directory}")

        self.vector_store = Chroma(
            persist_directory=persist_directory, embedding_function=self.embeddings
        )
        return self.vector_store

    def setup_qa_chain(self, temperature=0):
        """Set up the question-answering chain using the latest LangChain syntax."""
        if not self.vector_store:
            raise ValueError(
                "Vector store not initialized. Please create or load a vector store first."
            )

        llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,  # Changed back to openai_api_key
            temperature=temperature,
            model_name="gpt-3.5-turbo",
        )

        # Create prompt template
        template = """Answer the question based only on the following context:

        {context}

        Question: {question}
        
        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        # Create retrieval chain
        retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        # Create the RAG chain
        self.qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def query(self, question):
        """Query the RAG system with a question."""
        if not self.qa_chain:
            raise ValueError(
                "QA chain not initialized. Please run setup_qa_chain() first."
            )

        try:
            relevant_docs = self.retrive_relavant_vectors(question)
            self.print_retrieved_vecs_info(relevant_docs)
            answer = self.qa_chain.invoke(question)

            return {"answer": answer, "source_documents": relevant_docs}
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return None

    def retrive_relavant_vectors(self, question):
        # Get relevant documents for context
        relevant_docs = self.vector_store.similarity_search(question, k=4)
        # remove duplicates while preserving order
        # relevant_docs = list(dict.fromkeys(relevant_docs))
        relevant_docs = [doc for i, doc in enumerate(relevant_docs) if doc not in relevant_docs[:i]]
        return relevant_docs

    def print_retrieved_vecs_info(self, relevant_docs):
        print("Num Relevant documents:", len(relevant_docs))
        # print document lengths
        for doc in relevant_docs:
            print(f"Document length: {len(doc.page_content)}, Contents: {doc.page_content[:100]}")


def main():
    try:
        # Load environment variables
        load_dotenv()

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Initialize RAG application
        rag = RAGApplication(openai_api_key=api_key)

        # Set up your PDF directory
        pdf_directory = "../corpus/complete"

        # Load PDFs and create vector store
        texts = rag.load_pdfs(pdf_directory)
        print("Number of chunks:", len(texts))
        rag.create_vector_store(texts)

        # Setup QA chain
        rag.setup_qa_chain()

        # Example query
        question = "What is the issue pursuing 'great ideas'?"
        result = rag.query(question)

        if result:
            print("Answer:", result["answer"])
            print("\nSources:")
            for doc in result["source_documents"]:
                print(f"- {doc.metadata}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
