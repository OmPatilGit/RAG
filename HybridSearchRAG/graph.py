from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from operator import add
from langchain_core.documents import Document


from model import model
from retriever import get_hybrid_retriever
from prompts import GRADE_DOCS, FINAL_ANSWER

model = model()
retriever = get_hybrid_retriever()

class AgentState(TypedDict):
    messages : List[BaseMessage]
    question : str
    documents : List[Document]
    relevance : str
    result : str

def retriever_node(state : AgentState) -> AgentState:
    print("--- RETRIEVING DOCUMENTS ---")
    query = state['question']
    result = retriever.invoke(query)
    print(f"--- RETRIEVED {len(result)} DOCUMENTS ---")
    return {"documents" : result}

def grade_docs(state : AgentState) -> AgentState:
    print("--- GRADING DOCUMENTS ---")  
    question = state["question"]
    docs = state["documents"]
    prompt = GRADE_DOCS.format_prompt(question=question, documents=docs)

    result = model.invoke(prompt)
    if "yes" in result.lower():
        print("--- GRADE: DOCUMENTS ARE RELEVANT ---")
        return {"relevance": "YES"}
    else:
        print("--- GRADE: DOCUMENTS ARE NOT RELEVANT ---")
        return {"relevance": "NO"}
    
def generation(state : AgentState) -> AgentState:
    print("--- GENERATING ANSWER ---")

    query = state["question"]
    docs = state['documents']
    formatted_docs = "\n\n".join(doc.page_content for doc in docs)
    prompt = FINAL_ANSWER.format_prompt(context=formatted_docs, question=query)

    result = model.invoke(prompt)
    print("--- ANSWER GENERATED ---")

    return {"result" : result}

def should_continue(state : AgentState):
    relevance = state["relevance"]

    if "yes" in relevance.lower().strip():
        return "CONTINUE"
    return "END"

graph = StateGraph(AgentState)

graph.add_node("RETRIEVER NODE", retriever_node)
graph.add_node("GRADE NODE", grade_docs)
graph.add_node("GENERATION NODE", generation)

graph.add_edge(START, "RETRIEVER NODE")
graph.add_edge("RETRIEVER NODE", "GRADE NODE")

graph.add_conditional_edges(
    "GRADE NODE",
    should_continue,
    {
        "CONTINUE" : "GENERATION NODE",
        "END" : END
    }
)

graph.add_edge("GENERATION NODE", END)

app = graph.compile()

