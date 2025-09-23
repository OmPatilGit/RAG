import os 
from dotenv import load_dotenv
load_dotenv(dotenv_path=r"D:\Project\RAG\.env")

NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_CLIENT_ID = os.getenv("CLIENT_ID")
NEO4J_CLIENT_SECRET = os.getenv("CLIENT_SECRET")