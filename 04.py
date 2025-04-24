#Part 4: Adding Human-in-the-Loop (HITL) Assistance

import os
import getpass
import torch
from transformers import pipeline
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import Annotated
from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

tavily_api_key = os.getenv("TAVILY_API_KEY") or getpass.getpass("Enter your Tavily API key: ")

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

device = 0 if torch.cuda.is_available() else -1
dtype  = torch.float16 if device >= 0 else torch.float32

hf_pipe = pipeline(
    "text-generation",
    model="openai-community/gpt2",
    device=device,
    torch_dtype=dtype,
    do_sample=True,
    temperature=0.4,
    max_new_tokens=64,
)

search_tool = TavilySearchResults(tavily_api_key=tavily_api_key, max_results=1)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human operator."""
    resp = interrupt({"query": query})
    return resp["data"]

tools = [search_tool, human_assistance]

graph_builder = StateGraph(State)

def chatbot_node(state: State):
    last = state["messages"][-1]
    prompt = getattr(last, "content", getattr(last, "text", str(last)))
    gen = hf_pipe(prompt)[0]["generated_text"]
    return {"messages": [AIMessage(content=gen)]}

graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools",    ToolNode(tools=tools))

graph_builder.add_edge(START,    "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools",   "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

conversation_history: List[BaseMessage] = []

def stream_graph_updates(user_input: str):
    conversation_history.append(HumanMessage(content=user_input))
    init = {"messages": conversation_history}
    for event in graph.stream(init, stream_mode="values"):
        if "messages" in event:
            msg = event["messages"][-1]
            content = getattr(msg, "content", getattr(msg, "text", str(msg)))
            print("Assistant:", content)
            conversation_history.append(AIMessage(content=content))


if __name__ == "__main__":
    print("GPT-2 agent with web-search and HITL; improved memory. Type 'exit' to quit.")
    while True:
        q = input("User: ")
        if q.strip().lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        stream_graph_updates(q)
