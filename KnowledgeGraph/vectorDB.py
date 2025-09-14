from neo4j import GraphDatabase

import os 
from dotenv import load_dotenv
load_dotenv()

NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_CLIENT_ID = os.getenv("CLIENT_ID")
NEO4J_CLIENT_SECRET = os.getenv("CLIENT_SECRET")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_CLIENT_ID, NEO4J_CLIENT_SECRET))

try: 
    driver.verify_connectivity()
    print("Neo4j Connection Successfull !")
except Exception as e:
    print("Neo4j connection failed !")
    print("Error :", e)

