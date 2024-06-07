import os
import warnings
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.llms.openai import OpenAI

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
embeddings = OpenAIEmbeddings()

def get_text(file):
    if file.endswith(".pdf"):
        pdfreader = PdfReader(file)
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len
        )

        texts = text_splitter.split_text(raw_text)
        return texts


def embed_doc(file):
    document_search = FAISS.from_texts(get_text(file), embeddings)
    return document_search

def qna_pdf(document_search):
    qa_chain = load_qa_chain(OpenAI(), chain_type="stuff")
    while True:
        question = input("Ask a question: ")
        if question.lower().strip() == "exit":
            break
        else:
            docs = document_search.similarity_search(question)
            if docs:
                value = {"question": question, "input_documents": docs}
                output = qa_chain.run(value)
                print(output)

def load_csv(file):
    loader = CSVLoader(file_path=file)
    return loader

def create_csv_index(file):
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([load_csv(file)])
    return docsearch

def qna_csv(docsearch):
    qa_chain =RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
    while True:
        question = input("Ask a question: ")
        if question.lower().strip() == "exit":
            break
        else:
            value = {"question": question}
            output = qa_chain(value)
            print(output['result'])

if __name__ == "__main__":
    # FOR PDFs

    pdf_path = "PDFs/Ayamullah-Khan-FlowCV-Resume.pdf"
    document_search = embed_doc(pdf_path)
    qna_pdf(document_search)
    
    # FOR CSVs

    csv_path = "Excels/heart.csv"
    docsearch = create_csv_index(csv_path)
    qna_csv(docsearch)
