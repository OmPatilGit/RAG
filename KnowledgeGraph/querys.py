from KnowledgeGraph.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

from KnowledgeGraph.vectorDB import checkConnectivity

con = checkConnectivity(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

print(con)

