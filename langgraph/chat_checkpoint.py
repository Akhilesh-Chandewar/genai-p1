import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import HumanMessage

# Load .env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# ----------- Initialize Gemini LLM -----------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY,
)


# ----------- LangGraph State -----------
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ----------- Chatbot Node -----------
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ----------- Build Graph -----------
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


# ----------- Main CLI Chatbot -----------
def run_chatbot():
    print("🤖 ChatBot Ready! Type 'exit' to stop.\n")

    # MongoDB setup
    DB_URI = "mongodb://admin:admin@localhost:27017"

    with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:

        graph = graph_builder.compile(checkpointer=checkpointer)

        config = {
            "configurable": {"thread_id": "akhil"}
        }

        print("💾 Checkpointing enabled (MongoDB)\n")

        while True:
            user_msg = input("You: ").strip()

            if user_msg.lower() in ["exit", "quit", "bye"]:
                print("Bot: Goodbye!")
                break

            # Skip empty messages
            if not user_msg:
                continue

            # Invoke graph and get final state
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_msg)]}, 
                config
            )
            
            # Print only the last message (bot's response)
            bot_reply = result["messages"][-1]
            print(f"Bot: {bot_reply.content}\n")


# Run program
if __name__ == "__main__":
    run_chatbot()