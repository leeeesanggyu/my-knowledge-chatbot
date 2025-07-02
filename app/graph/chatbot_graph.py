import json

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from app.graph.print_utils import print_search_result
from app.velog.vector_store import VelogVectorStore

# 메모리 기반 사용자별 대화 이력 저장소
CHAT_HISTORY_STORE = {}


# 상태 정의
class GraphState(dict):
    chat_id: str
    question: str
    search_result: dict
    answer: str


# 벡터스토어 초기화
vector_store = VelogVectorStore()

# LLM 초기화 (LangGraph는 LangChain LLM 사용)
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

    # 메모리에서 대화 이력 불러오기 (없으면 빈 리스트)
    history = CHAT_HISTORY_STORE.get(chat_id, [])

    documents = search_result['documents'][0]
    metadatas = search_result['metadatas'][0]

    # context 구성
    context = ""
    for doc, meta in zip(documents, metadatas):
        title = meta.get('title', '제목 없음')
        url = meta.get('url', 'URL 없음')
        context += f"제목: {title}\n링크: {url}\n본문: {doc}\n\n"

    system_prompt = (
        "너는 나의 벨로그 블로그 포스팅을 참고하여 사용자의 질문에 답변하는 챗봇이다.\n"
        "다음은 관련 블로그 글이다:\n"
        f"{context}\n"
        "이 내용을 바탕으로 사용자의 질문에 최대한 상세히 답변해줘."
    )

    messages = [{"role": "system", "content": system_prompt}] + history
    messages.append({"role": "user", "content": question})

    print(f"message ::")
    print(json.dumps(messages, indent=2, ensure_ascii=False))

    response = llm.invoke(messages)

    # 이력 업데이트 → 메모리에 저장
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response.content})
    CHAT_HISTORY_STORE[chat_id] = history

    state['answer'] = response.content
    return state


# LangGraph 정의
graph = StateGraph(GraphState)

graph.add_node("search", search_node)
graph.add_node("generate_answer", answer_node)

# 흐름 연결
graph.set_entry_point("search")
graph.add_edge("search", "generate_answer")
graph.set_finish_point("generate_answer")

# 컴파일
runnable = graph.compile()

# print(runnable.get_graph().draw_mermaid())
