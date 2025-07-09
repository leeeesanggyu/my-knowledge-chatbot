import json
from typing import Dict, List

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.velog.vector_store import VelogVectorStore

# --- Constants ---
HISTORY_SUMMARY_THRESHOLD = 6
SEARCH_TOP_K = 3
SEARCH_MIN_SCORE = 0.6


# --- State Definition ---
class GraphState(dict):
    chat_id: str
    question: str
    search_result: dict
    answer: str


# --- Chatbot Graph ---
class ChatbotGraph:
    def __init__(self, llm_model: str = "gpt-3.5-turbo-0125"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.0)
        self.vector_store = VelogVectorStore()
        self.chat_history_store: Dict[str, List[Dict[str, str]]] = {}
        self.runnable = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(GraphState)

        graph.add_node("search", self.search_node)
        graph.add_node("router", lambda state: state)
        graph.add_node("generate_answer", self.answer_node)
        graph.add_node("fallback_response", self.fallback_response_node)
        graph.add_node("summarize", self.summarize_node)

        graph.set_entry_point("search")
        graph.add_edge("search", "router")
        graph.add_conditional_edges("router", self.route_by_search_result)
        graph.add_conditional_edges(
            "generate_answer",
            self.route_after_answer,
            {"summarize": "summarize", "next": END}
        )

        graph.add_edge("fallback_response", END)
        graph.add_edge("summarize", END)

        return graph.compile()

    def search_node(self, state: GraphState) -> GraphState:
        query = state['question']
        state['search_result'] = self.vector_store.search(
            query, top_k=SEARCH_TOP_K, min_score=SEARCH_MIN_SCORE
        )
        return state

    def answer_node(self, state: GraphState) -> GraphState:
        chat_id = state['chat_id']
        question = state['question']
        search_result = state['search_result']

        context = self._format_context(search_result)
        system_prompt = self._create_system_prompt(context)

        history = self.chat_history_store.get(chat_id, [])
        messages: List[BaseMessage] = [{"role": "system", "content": system_prompt}] + history
        messages.append({"role": "user", "content": question})

        response = self.llm.invoke(messages)
        self._update_chat_history(chat_id, question, response.content)

        state['answer'] = response.content
        return state

    def summarize_node(self, state: GraphState) -> GraphState:
        chat_id = state['chat_id']
        history = self.chat_history_store.get(chat_id, [])

        recent_history = history[-HISTORY_SUMMARY_THRESHOLD:]
        remaining_history = history[:-HISTORY_SUMMARY_THRESHOLD]

        prompt = [{"role": "system", "content": "다음은 사용자와 챗봇의 대화 내용이야. 핵심만 간단히 요약해줘."}] + recent_history
        summary_response = self.llm.invoke(prompt)
        summary = summary_response.content

        summarized_message = {"role": "system", "content": f"[요약된 이전 대화]\n{summary}"}
        self.chat_history_store[chat_id] = remaining_history + [summarized_message]

        print(f"\n[요약 완료] Chat ID: {chat_id}")
        print(json.dumps(summarized_message, indent=2, ensure_ascii=False))
        return state

    def fallback_response_node(self, state: GraphState) -> GraphState:
        state['answer'] = "죄송합니다, 질문하신 내용을 개발자 이상규의 Velog에서 찾을 수 없습니다."
        return state

    def route_by_search_result(self, state: GraphState) -> str:
        search_result = state["search_result"]
        has_result = len(search_result.get('documents', [[]])[0]) > 0
        return "generate_answer" if has_result else "fallback_response"

    def route_after_answer(self, state: GraphState) -> str:
        chat_id = state["chat_id"]
        history = self.chat_history_store.get(chat_id, [])
        return "summarize" if len(history) >= HISTORY_SUMMARY_THRESHOLD else "next"

    def _format_context(self, search_result: dict) -> str:
        documents = search_result['documents'][0]
        metadatas = search_result['metadatas'][0]
        context = ""
        for doc, meta in zip(documents, metadatas):
            title = meta.get('title', '제목 없음')
            url = meta.get('url', 'URL 없음')
            context += f"제목: {title}\n링크: {url}\n본문: {doc}\n\n"
        return context

    def _create_system_prompt(self, context: str) -> str:
        return (
            "너는 나의 벨로그 블로그 포스팅만 참고하여 사용자의 질문에 답변하는 챗봇이야.\n"
            "절대 너의 지식이나 외부 정보에 근거해 답변하지 마.\n"
            "다음은 관련 블로그 글이야:\n"
            f"{context}\n"
            "이 내용을 바탕으로 사용자의 질문에 최대한 상세히 답변해줘."
        )

    def _update_chat_history(self, chat_id: str, question: str, answer: str):
        history = self.chat_history_store.get(chat_id, [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        self.chat_history_store[chat_id] = history
        print(f"\n\n[Chat History Updated] Chat ID: {chat_id}")
        print(json.dumps(self.chat_history_store[chat_id], indent=2, ensure_ascii=False))

    def invoke(self, state: dict, debug: bool = False):
        return self.runnable.invoke(state, {"recursion_limit": 10}, debug=debug)


# --- Singleton Instance ---
chatbot_graph_instance = ChatbotGraph()
runnable = chatbot_graph_instance.invoke
