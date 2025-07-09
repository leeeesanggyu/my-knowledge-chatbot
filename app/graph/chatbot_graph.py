import json
from typing import Dict, List

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.velog.vector_store import VelogVectorStore

# Constants: RAG 파이프라인 및 대화 요약에 사용될 상수
HISTORY_SUMMARY_THRESHOLD = 6  # 대화 요약을 시작할 대화 기록의 수
SEARCH_TOP_K = 3  # 벡터 저장소에서 검색할 문서의 수
SEARCH_MIN_SCORE = 0.6  # 검색 결과로 인정할 최소 유사도 점수


class GraphState(dict):
    """
    그래프의 상태를 나타냅니다.

    Attributes:
        chat_id (str): 현재 채팅 세션의 ID
        question (str): 사용자의 질문
        search_result (dict): 벡터 저장소 검색 결과
        answer (str): 생성된 답변
    """
    chat_id: str
    question: str
    search_result: dict
    answer: str


class ChatbotGraph:
    """
    RAG (Retrieval-Augmented Generation) 챗봇의 전체 워크플로우를 관리합니다.
    LangGraph를 사용하여 검색, 답변 생성, 대화 요약 등의 단계를 그래프로 구성합니다.
    """

    def __init__(self, llm_model: str = "gpt-3.5-turbo-0125"):
        """
        ChatbotGraph를 초기화합니다.

        Args:
            llm_model (str): 사용할 LLM 모델의 이름
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=0.0)
        self.vector_store = VelogVectorStore()
        self.chat_history_store: Dict[str, List[Dict[str, str]]] = {}  # 채팅 ID별 대화 기록 저장
        self.runnable = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        LangGraph를 사용하여 챗봇의 워크플로우 그래프를 구성합니다.
        """
        graph = StateGraph(GraphState)

        # 노드 정의
        graph.add_node("search", self.search_node)  # 문서 검색
        graph.add_node("router", lambda state: state)  # 검색 결과에 따라 분기
        graph.add_node("generate_answer", self.answer_node)  # 답변 생성
        graph.add_node("fallback_response", self.fallback_response_node)  # 검색 실패 시 응답
        graph.add_node("summarize", self.summarize_node)  # 대화 요약

        # 엣지 및 진입점/종료점 정의
        graph.set_entry_point("search")
        graph.add_edge("search", "router")
        graph.add_conditional_edges("router", self.route_by_search_result)  # 검색 결과에 따라 분기
        graph.add_conditional_edges(
            "generate_answer",
            self.route_after_answer,
            {"summarize": "summarize", "next": END}  # 답변 후 대화 기록 길이에 따라 분기
        )
        graph.add_edge("fallback_response", END)
        graph.add_edge("summarize", END)

        return graph.compile()

    def search_node(self, state: GraphState) -> GraphState:
        """
        사용자의 질문을 기반으로 벡터 저장소에서 관련 문서를 검색합니다.
        """
        query = state['question']
        state['search_result'] = self.vector_store.search(
            query, top_k=SEARCH_TOP_K, min_score=SEARCH_MIN_SCORE
        )
        return state

    def answer_node(self, state: GraphState) -> GraphState:
        """
        검색된 컨텍스트와 대화 기록을 사용하여 사용자의 질문에 답변을 생성합니다.
        """
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
        """
        대화 기록이 특정 임계값을 초과하면, 대화 내용을 요약하여 관리합니다.
        """
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
        """
        관련 문서를 찾지 못했을 때 사용자에게 전달할 기본 응답을 설정합니다.
        """
        state['answer'] = "죄송합니다, 질문하신 내용을 개발자 이상규의 Velog에서 찾을 수 없습니다."
        return state

    def route_by_search_result(self, state: GraphState) -> str:
        """
        검색 결과의 유무에 따라 다음 노드를 결정합니다.
        결과가 있으면 'generate_answer', 없으면 'fallback_response'로 분기합니다.
        """
        search_result = state["search_result"]
        has_result = len(search_result.get('documents', [[]])[0]) > 0
        return "generate_answer" if has_result else "fallback_response"

    def route_after_answer(self, state: GraphState) -> str:
        """
        답변 생성 후, 대화 기록의 길이에 따라 다음 노드를 결정합니다.
        임계값을 넘으면 'summarize', 아니면 'next'(종료)로 분기합니다.
        """
        chat_id = state["chat_id"]
        history = self.chat_history_store.get(chat_id, [])
        return "summarize" if len(history) >= HISTORY_SUMMARY_THRESHOLD else "next"

    def _format_context(self, search_result: dict) -> str:
        """
        검색된 문서들을 LLM에 전달할 컨텍스트 문자열로 포맷합니다.
        """
        documents = search_result['documents'][0]
        metadatas = search_result['metadatas'][0]
        context = ""
        for doc, meta in zip(documents, metadatas):
            title = meta.get('title', '제목 없음')
            url = meta.get('url', 'URL 없음')
            context += f"제목: {title}\n링크: {url}\n본문: {doc}\n\n"
        return context

    def _create_system_prompt(self, context: str) -> str:
        """
        LLM에 전달할 시스템 프롬프트를 생성합니다.
        """
        return (
            "너는 나의 벨로그 블로그 포스팅만 참고하여 사용자의 질문에 답변하는 챗봇이야.\n"
            "절대 너의 지식이나 외부 정보에 근거해 답변하지 마.\n"
            "다음은 관련 블로그 글이야:\n"
            f"{context}\n"
            "이 내용을 바탕으로 사용자의 질문에 최대한 상세히 답변해줘."
        )

    def _update_chat_history(self, chat_id: str, question: str, answer: str):
        """
        질문과 답변을 대화 기록에 추가합니다.
        """
        history = self.chat_history_store.get(chat_id, [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        self.chat_history_store[chat_id] = history
        print(f"\n\n[Chat History Updated] Chat ID: {chat_id}")
        print(json.dumps(self.chat_history_store[chat_id], indent=2, ensure_ascii=False))

    def invoke(self, state: dict, debug: bool = False):
        """
        컴파일된 그래프를 실행합니다.

        Args:
            state (dict): 초기 상태 (chat_id, question 포함)
            debug (bool): 디버그 모드 활성화 여부
        """
        return self.runnable.invoke(state, {"recursion_limit": 10}, debug=debug)


# ChatbotGraph의 단일 인스턴스를 생성하여 사용
chatbot_graph_instance = ChatbotGraph()
runnable = chatbot_graph_instance.invoke
