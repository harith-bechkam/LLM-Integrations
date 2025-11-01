import time
from typing import List, Dict
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnableWithMessageHistory, Runnable
from langchain_core.messages import AIMessage, HumanMessage

# ------------------------------
# Memory / History Management
# ------------------------------
def build_memory_from_messages(messages: List[dict]) -> ChatMessageHistory:
    """Rebuild conversation history for a chat session."""
    history = ChatMessageHistory()
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if not role or not content:
            continue
        if role == "user":
            history.add_user_message(content)
        elif role == "assistant":
            history.add_ai_message(content)
    return history

# ------------------------------
# Custom Runnables
# ------------------------------
class PromptRunnable(Runnable):
    """Wraps a ChatPromptTemplate as a Runnable."""

    def __init__(self, prompt: ChatPromptTemplate):
        self.prompt = prompt

    def invoke(self, input, *args, **kwargs) -> str:
        """
        Accept input as:
        - dict
        - list containing dicts (RunnableSequence passes a list)
        """
        if isinstance(input, list):
            if len(input) == 1 and isinstance(input[0], dict):
                input = input[0]
            else:
                input = {"input": " ".join(str(i) for i in input)}

        if not isinstance(input, dict):
            raise TypeError(f"Expected input to be dict, got {type(input)}")

        # format_prompt will receive 'history' automatically from RunnableWithMessageHistory
        formatted = self.prompt.format_prompt(**input)
        return "\n".join([msg.content for msg in formatted.to_messages()])


class LLMRunnable(Runnable):
    """Wraps an LLM as a Runnable."""

    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input: str, *args, **kwargs) -> str:
        response = self.llm.generate([HumanMessage(content=input)])
        return response.generations[0][0].text

# ------------------------------
# Chain Creation
# ------------------------------
def create_assistant_chain_with_memory(llm, messages: List[dict]):
    """
    Create a chat chain that maintains message history.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that provides clear, structured, and concise answers."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}")
    ])

    prompt_runnable = PromptRunnable(prompt)
    llm_runnable = LLMRunnable(llm)

    # RunnableWithMessageHistory automatically injects history into the prompt
    chain = RunnableWithMessageHistory(
        runnable=RunnableSequence(prompt_runnable, llm_runnable),
        get_session_history=lambda session_id=None: build_memory_from_messages(messages)
    )

    return chain

# ------------------------------
# Insert Messages in MongoDB
# ------------------------------
async def insert_message(chats_coll, chat_id: str, role: str, content: str):
    """Insert a chat message into MongoDB and update timestamp."""
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
