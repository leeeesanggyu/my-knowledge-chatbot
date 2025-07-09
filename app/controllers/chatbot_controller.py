from fastapi import APIRouter, HTTPException
from app.models.chatbot_models import ChatRequest, ChatResponse
from app.graph.chatbot_graph import runnable

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        state = {
            "chat_id": request.chat_id,
            "question": request.question
        }

        return runnable.invoke(state, debug=False)
    except Exception as e:
        print("LangGraph 실행 실패:", e)
        raise HTTPException(status_code=500, detail=str(e))
