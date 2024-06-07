import os
import warnings
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
embeddings = OpenAIEmbeddings()

def get_text(pdf):
    pdfreader = PdfReader(pdf)
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

def embed_pdf(pdf):
    document_search = FAISS.from_texts(get_text(pdf), embeddings)
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
            else:
                print("No relevant documents found.")

if __name__ == "__main__":
    pdf_path = "PDFs/Ayamullah-Khan-FlowCV-Resume.pdf"
    document_search = embed_pdf(pdf_path)
    qna_pdf(document_search)
