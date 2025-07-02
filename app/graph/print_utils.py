
def print_search_result(search_result):
    documents = search_result.get('documents', [[]])[0]
    metadatas = search_result.get('metadatas', [[]])[0]
    ids = search_result.get('ids', [[]])[0]
    distances = search_result.get('distances', [[]])[0]


    for i, (doc, meta, id, distance) in enumerate(zip(documents, metadatas, ids, distances), 1):
        print(f"============ 🔎 검색 결과 {i} ============")
        print(f"유사도 점수: {1 - distance:.2f}")
        print(f"ID: {id}")
        print(f"Title: {meta.get('title', '제목없음')}")
        print(f"Content (앞부분): {doc[:300]}...\n")  # 앞 300자만 미리보기로 출력
        print(f"========================================")