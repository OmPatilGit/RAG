from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List ,Annotated
from operator import add
from langchain_core.documents import Document
from IPython.display import Image, display

from HybridSearchRAG.model import GenModel
from HybridSearchRAG.retriever import get_hybrid_retriever
from HybridSearchRAG.prompts import GRADE_DOCS, FINAL_ANSWER

model = GenModel()
retriever = get_hybrid_retriever()

class AgentState(TypedDict):
    messages : Annotated[List[BaseMessage], add]
    question : str
    documents :Annotated[ List[Document], add]
    relevance : str
    result : str

def retriever_node(state : AgentState) -> AgentState:
    print("--- RETRIEVING DOCUMENTS ---")
    if state['question']:
        query = state['question']
    else:
        query = state['messages'][-1].content
    result = retriever.invoke(query)
    print(f"--- RETRIEVED {len(result)} DOCUMENTS ---")
    return {"documents" : result}

def grade_docs(state : AgentState) -> AgentState:
    print("--- GRADING DOCUMENTS ---")  
    question = state["question"]
    docs = state["documents"]
    prompt = GRADE_DOCS.format_prompt(question=question, documents=docs)

    result = model.invoke(prompt)
    if "yes" in result.content.lower():
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


if __name__ == "__main__":
    try:
        with open("graph_visualization.png", "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print("Graph visualization saved to graph_visualization.png")
    except Exception as e:
        print(f"Error saving graph visualization: {e}")
