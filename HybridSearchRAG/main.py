from langgraph.graph import StateGraph, START, END
from HybridSearchRAG.graph import (
    AgentState, 
    retriever_node, 
    grade_docs, 
    generation, 
    should_continue
    )
from langchain_core.messages import HumanMessage, AIMessage

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

# ...existing code...

initial_state = {
    "messages": [],
    "question": "",
    "documents": [],
    "relevance": "",
    "result": ""
}

print("Chatbot is ready! Type 'quit' or 'exit' to end.")
state = initial_state.copy()
while True:
    query = input("USER : ")
    if query.lower().strip() in ['bye', 'exit', 'quit']:
        print("AI : Goodbye!")
        break

    state["messages"].append(HumanMessage(content=query))
    state["question"] = query
    result_state = app.invoke(state)
    state["messages"].append(result_state["result"])  
    print(f"AI : {result_state['result'].content}")
    state.update(result_state)