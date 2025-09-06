from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("BASE_URL")

def GenModel(name : str = "gpt-oss-120b", 
             temperature : float = 1.0):
    
    llm = ChatOpenAI(
        model=name,
        api_key=OPENROUTER_API_KEY,
        temperature=temperature,
        base_url=BASE_URL,
        default_headers={
        "HTTP-Referer": "http://localhost",   
        "X-Title": "Agent for project"
        }
    )

    return llm