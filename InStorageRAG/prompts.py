from langchain_core.prompts import PromptTemplate

INSTORAGE_RAG = PromptTemplate(template="""
## Role : You are the best and helpful assistant
## Task : You have to repond users question based on the context.
## Context : {context}
## Question : {question}
## Answer : """,
input_variables=['context', 'question'])