import os
import time
from typing import List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Import your existing LangChain helper
from helpers import create_assistant_chain_with_memory, insert_message

load_dotenv()

# ---------- Setup LLM ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# ---------- Define shared state ----------
from typing import TypedDict

class ChatState(TypedDict):
    messages: List[Dict[str, Any]]


# ---------- Agent 1: ChatAgent ----------
def chat_agent_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    user_msgs = [m for m in messages if m["role"] == "user"]
    if not user_msgs:
        return state  # nothing to respond to
    user_msg = user_msgs[-1]["content"]

    # Use your existing LangChain chatbot logic
    chain = create_assistant_chain_with_memory(llm, messages)
    assistant_reply = chain.predict(input=user_msg)

    new_messages = messages + [{"role": "assistant", "content": assistant_reply, "ts": time.time()}]
    return {"messages": new_messages}


# ---------- Agent 2: SummarizerAgent ----------
def summarizer_agent_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    if not assistant_msgs:
        return state  # nothing to summarize
    last_assistant = assistant_msgs[-1]["content"]

    summary = f"Summary: {last_assistant[:80]}..."
    new_messages = messages + [{"role": "assistant", "content": summary, "ts": time.time()}]
    return {"messages": new_messages}


# ---------- Conditional Edge ----------
def should_summarize_edge(state: ChatState) -> str:
    messages = state["messages"]
    user_msgs = [m for m in messages if m["role"] == "user"]
    if not user_msgs:
        return END
    last_user_msg = user_msgs[-1]["content"]

    if "summary" in last_user_msg.lower():
        return "SummarizerAgent"
    return END


# ---------- Build LangGraph ----------
graph = StateGraph(ChatState)
graph.add_node("ChatAgent", chat_agent_node)
graph.add_node("SummarizerAgent", summarizer_agent_node)

graph.add_edge(START, "ChatAgent")
graph.add_conditional_edges("ChatAgent", should_summarize_edge, {"SummarizerAgent": "SummarizerAgent"})
graph.add_edge("SummarizerAgent", END)

# Compile graph
multi_agent_app = graph.compile()


# ---------- Helper to run the workflow ----------
def run_multi_agent_workflow(user_message: str, history: List[Dict[str, Any]] = None):
    messages = history or []
    messages.append({"role": "user", "content": user_message, "ts": time.time()})
    state: ChatState = {"messages": messages}
    final_state = multi_agent_app.run(state)
    return final_state["messages"]
