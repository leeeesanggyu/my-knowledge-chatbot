from fastapi import FastAPI
from app.controllers import chatbot_controller

app = FastAPI(
    title="Blog Chatbot API",
    version="0.1.0"
)

app.include_router(chatbot_controller.router, prefix="/api/chatbot")
