from langchain_core.prompts import PromptTemplate

GRADE_DOCS = PromptTemplate(template="""
## Role : You are a world class relevance grading assistant.
## Task : You have to grade the question and the documents based on their relevance.
## Output Format : If question and documents are related just say "YES", 
                   If question and documents have no relevance just say "NO" 
## Question : {question}
## Documents : {documents}
## Result : """,
input_variables=['question', 'documents'])


FINAL_ANSWER = PromptTemplate(template="""
## Role : You are helpfull assistant. You are being used as a Hybrid Search RAG Assistant.
## Task : Generate a answer based on the context you have for the given question.
## Context : {context}
## Question : {question}
## Answer : """,
input_variables=['context', 'question'])
