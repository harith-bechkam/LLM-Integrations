import os
import time
from uuid import uuid4
from typing import List
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient

# Pydantic models
from models import NewChatResponse, ChatListItem, MessageIn, MessageOut, SendMessageResponse

# Helpers
from helpers import create_assistant_chain_with_memory, insert_message

# LangChain OpenAI
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

# ---------- Config ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "chatbot")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORT = int(os.getenv("PORT", 8000))

# ---------- MongoDB setup ----------
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
chats_coll = db["chats"]

# ---------- LLM ----------
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# ---------- FastAPI ----------
app = FastAPI(title="LangChain Chatbot with OpenAI + MongoDB")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- API Endpoints ----------

@app.post("/chats", response_model=NewChatResponse)
async def create_chat(payload: MessageIn):
    chat_id = str(uuid4())
    now = time.time()
    title = payload.title or "New chat"
    await chats_coll.insert_one({
        "_id": chat_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": []
    })
    return {"chat_id": chat_id, "created_at": now}


@app.get("/chats", response_model=List[ChatListItem])
async def list_chats():
    cursor = chats_coll.find({}, {"messages": {"$slice": -1}, "title": 1, "updated_at": 1})
    items = []
    async for doc in cursor:
        last_msg = doc["messages"][-1]["content"] if doc.get("messages") else None
        items.append({
            "chat_id": doc["_id"],
            "title": doc.get("title", "New chat"),
            "last_message": last_msg,
            "updated_at": doc.get("updated_at", doc.get("created_at"))
        })
    items.sort(key=lambda x: x["updated_at"] or 0, reverse=True)
    return items


@app.get("/chats/{chat_id}", response_model=List[MessageOut])
async def get_chat(chat_id: str):
    doc = await chats_coll.find_one({"_id": chat_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Chat not found")
    messages = doc.get("messages", [])
    return [MessageOut(**m) for m in messages]


@app.post("/chats/{chat_id}/message", response_model=SendMessageResponse)
async def send_message(chat_id: str, payload: MessageIn):
    doc = await chats_coll.find_one({"_id": chat_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Chat not found")

    if payload.title:
        await chats_coll.update_one({"_id": chat_id}, {"$set": {"title": payload.title}})

    messages = doc.get("messages", [])
    user_text = payload.message

    # Build chain with memory
    chain = create_assistant_chain_with_memory(llm, messages)

    # Generate assistant response
    assistant_text = chain.predict(input=user_text)

    # Save messages
    await insert_message(chats_coll, chat_id, "user", user_text)
    await insert_message(chats_coll, chat_id, "assistant", assistant_text)

    # Return updated messages
    updated_doc = await chats_coll.find_one({"_id": chat_id})
    updated_messages = updated_doc.get("messages", [])

    return {
        "assistant": assistant_text,
        "chat_id": chat_id,
        "messages": updated_messages
    }


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    res = await chats_coll.delete_one({"_id": chat_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"deleted": True}
