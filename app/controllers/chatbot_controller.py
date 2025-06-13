from fastapi import APIRouter, HTTPException
from app.models.chatbot_models import ChatRequest, ChatResponse
from app.services.chatbot_service import generate_answer

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = generate_answer(request.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))