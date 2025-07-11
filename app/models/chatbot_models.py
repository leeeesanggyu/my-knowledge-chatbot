from pydantic import BaseModel

class ChatRequest(BaseModel):
    chat_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
