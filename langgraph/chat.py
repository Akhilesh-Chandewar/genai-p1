import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY,
)

# -------------------------------
# LangGraph State
# -------------------------------


class State(TypedDict):
    messages: Annotated[list, add_messages]


# -------- Chatbot Node ---------


def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# -------- Build Graph --------

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

# Path: START → chatbot → END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# -------------------------------
# Chat Loop
# -------------------------------


def run_chatbot():
    print("🤖 ChatBot Ready! Type 'exit' to stop.\n")
    state = {"messages": []}

    while True:
        user_msg = input("You: ")

        if user_msg.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye!")
            break

        state["messages"].append(user_msg)

        output = graph.invoke(state)
        state["messages"] = output["messages"]

        bot_reply = state["messages"][-1]

        if hasattr(bot_reply, "content"):
            print("Bot:", bot_reply.content)
        else:
            print("Bot:", bot_reply)


# Run program
if __name__ == "__main__":
    run_chatbot()
