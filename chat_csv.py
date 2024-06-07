import os
import warnings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.llms.openai import OpenAI

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

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
    csv_path = "Excels/heart.csv"
    docsearch = create_csv_index(csv_path)
    qna_csv(docsearch)