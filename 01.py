#Part 1: Set up the chatbot with LangGraph
import torch
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated    
from typing_extensions import TypedDict


device = 0 if torch.cuda.is_available() else -1
dtype  = torch.float16 if device >= 0 else torch.float32

SYSTEM_DESCRIPTION = "You are a helpful AI assistant that answers questions clearly and concisely."


hf_pipe = pipeline(
    "text-generation",
    model="gpt2-medium",
    device=device,
    torch_dtype=dtype,
    max_length=256,
    truncation=True,
    do_sample=True,
    temperature=0.8,
)
llm = HuggingFacePipeline(pipeline=hf_pipe)


class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [ llm.invoke(state["messages"]) ]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    messages = [
        {"role": "system",  "content": SYSTEM_DESCRIPTION},
        {"role": "user",    "content": user_input}
    ]

    for event in graph.stream({"messages": messages}):
        for value in event.values():
            reply = value["messages"][-1]
            if isinstance(reply, dict):
                print("Assistant:", reply.get("content", reply))
            else:
                print("Assistant:", reply)

if __name__ == "__main__":
    print(f"Using {'GPU' if device>=0 else 'CPU'} for inference.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
