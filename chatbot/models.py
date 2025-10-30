from pydantic import BaseModel
from typing import List, Optional


class NewChatResponse(BaseModel):
    chat_id: str
    created_at: float


class ChatListItem(BaseModel):
    chat_id: str
    title: str
    last_message: Optional[str]
    updated_at: float


class MessageIn(BaseModel):
    message: str
    title: Optional[str] = None


class MessageOut(BaseModel):
    role: str
    content: str
    ts: float


class SendMessageResponse(BaseModel):
    assistant: str
    chat_id: str
    messages: List[MessageOut]
