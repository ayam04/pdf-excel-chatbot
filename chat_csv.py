import os
import warnings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, Tool

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
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
    
    tools = [
        Tool(
            name="Answer Questions",
            func=lambda x: qa_chain({"question": x})['result'],
            description="Use this tool to answer questions based on the CSV data."
        )
    ]
    
    agent = initialize_agent(
        tools=tools,
        llm=OpenAI(),
        agent="zero-shot-react-description"
    )
    
    while True:
        command = input("Ask a question or give a command: ")
        if command.lower().strip() == "exit":
            break
        else:
            result = agent.run(command)
            print(result)

if __name__ == "__main__":
    csv_path = "Excels/heart.csv"
    docsearch = create_csv_index(csv_path)
    qna_csv(docsearch)
