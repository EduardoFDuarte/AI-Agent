# Part 2: Adding a Web-Search Tool


import os
import getpass
import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import Annotated
from typing import TypedDict, List, Union
from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    tavily_api_key = getpass.getpass("Enter your Tavily API key: ")


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


device = 0 if torch.cuda.is_available() else -1
dtype  = torch.float16 if device >= 0 else torch.float32

hf_pipe = pipeline(
    "text-generation",
    model="openai-community/gpt2",
    device=device,
    torch_dtype=dtype,
    max_length=256,
    truncation=True,
    do_sample=True,
    temperature=0.5,
)
llm = HuggingFacePipeline(pipeline=hf_pipe)


web_search = TavilySearchResults(tavily_api_key=tavily_api_key, max_results=1)
tools     = [web_search]


graph_builder = StateGraph(State)

def chatbot_node(state: State):
    return {"messages": [ llm.invoke(state["messages"]) ]}

graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools",    ToolNode(tools=tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages":[{"role":"user","content":user_input}]}):
        for out in event.values():
            msg = out["messages"][-1]
            if isinstance(msg, str):
                text = msg
            else:
                text = getattr(msg, "text", None) or getattr(msg, "content", None) or str(msg)
            print("Assistant:", text)


if __name__ == "__main__":
    print("GPT-2 agent with web-search tools. Type 'exit' to quit.")
    while True:
        q = input("User: ")
        if q.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        stream_graph_updates(q)
