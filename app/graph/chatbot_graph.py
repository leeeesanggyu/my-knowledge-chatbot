import json

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from app.velog.vector_store import VelogVectorStore

# 메모리 기반 사용자별 대화 이력 저장소
CHAT_HISTORY_STORE = {}


# 상태 정의
class GraphState(dict):
    chat_id: str
    question: str
    search_result: dict
    answer: str


vector_store = VelogVectorStore()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.0)


# 검색 노드
def search_node(state: GraphState):
    query = state['question']
    search_result = vector_store.search(query, top_k=3, min_score=0.6)
    state['search_result'] = search_result
    return state


# LLM 답변 노드
def answer_node(state: GraphState):
    chat_id = state['chat_id']
    question = state['question']
    search_result = state['search_result']

    documents = search_result['documents'][0]
    metadatas = search_result['metadatas'][0]

    # context 구성
    context = ""
    for doc, meta in zip(documents, metadatas):
        title = meta.get('title', '제목 없음')
        url = meta.get('url', 'URL 없음')
        context += f"제목: {title}\n링크: {url}\n본문: {doc}\n\n"

    system_prompt = (
        "너는 나의 벨로그 블로그 포스팅만 참고하여 사용자의 질문에 답변하는 챗봇이야.\n"
        "절대 너의 지식이나 외부 정보에 근거해 답변하지 마.\n"
        "다음은 관련 블로그 글이야:\n"
        f"{context}\n"
        "이 내용을 바탕으로 사용자의 질문에 최대한 상세히 답변해줘."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # 메모리에서 대화 이력 불러오기 (없으면 빈 리스트)
    history = CHAT_HISTORY_STORE.get(chat_id, [])

    messages += history
    messages.append({"role": "user", "content": question})

    response = llm.invoke(messages)

    # 이력 업데이트 → 메모리에 저장
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response.content})
    CHAT_HISTORY_STORE[chat_id] = history
    print("\n\n[Chat history]")
    print(json.dumps(CHAT_HISTORY_STORE, indent=2, ensure_ascii=False))

    state['answer'] = response.content
    return state


# 요약 노드
def summarize_node(state: GraphState):
    chat_id = state['chat_id']
    history = CHAT_HISTORY_STORE.get(chat_id, [])

    # 최근 10개 메시지를 요약 대상으로 사용
    recent = history[-6:]
    remaining = history[:-6]

    # 요약 프롬프트 생성
    prompt = [{"role": "system", "content": "다음은 사용자와 챗봇의 대화 내용이야. 핵심만 간단히 요약해줘."}] + recent
    response = llm.invoke(prompt)

    summary = response.content

    # 요약 메시지를 system role로 추가
    summarized_message = {
        "role": "system",
        "content": f"[요약된 이전 대화]\n{summary}"
    }

    # 기존 이력에서 recent 제거, 요약 메시지 추가
    new_history = remaining + [summarized_message]

    # 다시 저장
    CHAT_HISTORY_STORE[chat_id] = new_history

    print(f"[요약 완료]")
    print(json.dumps(summarized_message, indent=2, ensure_ascii=False))

    return state


def fallback_response_node(state: GraphState):
    state['answer'] = "죄송합니다, 질문하신 내용을 개발자 이상규의 Velog에서 찾을 수 없습니다."
    return state


# LangGraph 정의
graph = StateGraph(GraphState)

graph.add_node("search", search_node)
graph.add_node("router", lambda state: state)
graph.add_node("generate_answer", answer_node)
graph.add_node("fallback_response", fallback_response_node)
graph.add_node("summarize", summarize_node)


def route_by_search_result(state: GraphState):
    search_result = state["search_result"]
    has_result = len(search_result.get('documents', [[]])[0]) > 0
    return "generate_answer" if has_result else "fallback_response"


# History 요약 분기 함수
def route_after_answer(state: GraphState):
    chat_id = state["chat_id"]
    history = CHAT_HISTORY_STORE.get(chat_id, [])
    return "summarize" if len(history) >= 6 else "next"


# 간선 연결
graph.set_entry_point("search")
graph.add_edge("search", "router")  # router는 상태 그대로 넘김
graph.add_conditional_edges("router", route_by_search_result)  # 조건 분기
graph.add_conditional_edges("generate_answer", route_after_answer, {
    "summarize": "summarize",
    "next": "__end__"
})

# 종료 지점
graph.set_finish_point("generate_answer")
graph.set_finish_point("fallback_response")
graph.set_finish_point("summarize")

# 컴파일
runnable = graph.compile()

# print(runnable.get_graph().draw_mermaid())
