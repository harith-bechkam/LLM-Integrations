import time
from typing import List
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory  # updated import


def build_memory_from_messages(messages: List[dict]) -> ConversationBufferMemory:
    """
    Build LangChain memory from stored chat messages
    """
    chat_history = ChatMessageHistory()
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            chat_history.add_user_message(content)
        else:
            chat_history.add_ai_message(content)
    memory = ConversationBufferMemory(chat_memory=chat_history, return_messages=True)
    return memory


async def insert_message(chats_coll, chat_id: str, role: str, content: str):
    """
    Insert a message into MongoDB chat collection
    """
    ts = time.time()
    await chats_coll.update_one(
        {"_id": chat_id},
        {
            "$push": {"messages": {"role": role, "content": content, "ts": ts}},
            "$set": {"updated_at": ts},
        },
        upsert=False,
    )
    return {"role": role, "content": content, "ts": ts}


def create_assistant_chain_with_memory(llm, messages) -> ConversationChain:
    """
    Create a LangChain ConversationChain with preloaded memory
    """
    memory = build_memory_from_messages(messages)
    chain = ConversationChain(llm=llm, memory=memory, verbose=False)
    return chain
