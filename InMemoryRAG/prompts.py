from langchain.prompts import PromptTemplate

INMEMORY_RETRIEVER = PromptTemplate(template="""
Role : You are a RAG assistant.
Task : Answer the question based on the context you have.
Context : {context}
Question : {question} 
Answer : """,
input_variables=['context', 'question'])