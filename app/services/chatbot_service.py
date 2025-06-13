from app.models.chatbot_models import ChatResponse

def generate_answer(question: str) -> ChatResponse:
    answer = f"당신의 질문은 '{question}' 입니다. (여기에 LangChain 답변이 들어갑니다)"
    return ChatResponse(answer=answer)
