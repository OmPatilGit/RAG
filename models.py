import os
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
BASE_URL = os.getenv("BASE_URL")

from langchain_openai import ChatOpenAI
from typing import List
from langchain_ollama import OllamaEmbeddings


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


def EmbModel(text : List[str],
             name : str = "mxbai-embed-large:335m",
             document : bool = False):
    embeddings = OllamaEmbeddings(model=name)
    if document:
        return embeddings.embed_documents(texts=text)
    else:
        return embeddings.embed_query(text[0])
    