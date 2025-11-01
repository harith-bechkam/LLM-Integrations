import time
from typing import List, Dict
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder  # if you need history insertion
# If you need message‑type prompt templates:
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain.memory.buffer import ConversationBufferMemory
pip install "langchain<1.0"

from langchain.chains import LLMChain


# ------------------------------
# Memory / History Management
# ------------------------------
def build_chat_history(messages: List[dict]) -> str:
    """
    Convert MongoDB messages to a single string for context
    (optional, if you want to pass history manually)
    """
    chat_history = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role and content:
            prefix = "User:" if role == "user" else "Assistant:"
            chat_history.append(f"{prefix} {content}")
    return "\n".join(chat_history)


# ------------------------------
# Chain Creation
# ------------------------------
def create_llm_chain_with_memory(llm, messages: List[dict]):
    """
    Create a simple LLMChain with ConversationBufferMemory.
    """
    # Memory automatically stores the conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Load existing messages into memory
    for m in messages:
        if m["role"] == "user":
            memory.chat_memory.add_user_message(m["content"])
        elif m["role"] == "assistant":
            memory.chat_memory.add_ai_message(m["content"])

    # Create prompt template
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a helpful AI assistant that provides clear, structured, and concise answers."
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    return chain, memory
