import os
import warnings
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

def load_csv(file):
    filename = os.path.basename(file)
    if file.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.endswith(".xlsx"):
        df = pd.read_excel(file)

    engine = create_engine(f"sqlite:///{filename}.db")
    df.to_sql(f"{filename}", con=engine, index=False, if_exists="replace")
    db = SQLDatabase(engine)
    return db

def create_agent(file):
    agent = create_sql_agent(
        db=load_csv(file),
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        agent="openai-tools",
        verbose=False
    )
    return agent

def qna_csv(agent):
    while True:
        command = input("Ask a question: ")
        if command.lower().strip() == "exit":
            break
        else:
            result = agent.run(command)
            print(result)

if __name__ == "__main__":
    csv_path = "pdf-excel-chatbot/Excels/Day 5 survey with email - values.csv"
    agent = create_agent(csv_path)
    qna_csv(agent)