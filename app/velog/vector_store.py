import chromadb
import dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os

from app.graph.print_utils import print_search_result

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_DIR = './chroma_db'


class VelogVectorStore:

    def __init__(self):
        self.embedding_func = OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-ada-002"
        )
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.client.get_or_create_collection("velog-posts", embedding_function=self.embedding_func)

    def add_post(self, post_id, content, metadata):
        self.collection.add(
            ids=[post_id],
            documents=[content],
            metadatas=[metadata]
        )
        print(f"✅ {metadata['title']} 저장 완료")

    def search(self, query, top_k=3, min_score: float = 0.6):
        """
        min_score: 0 ~ 1 사이의 값 (1에 가까울수록 더 유사한 결과만 반환)
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        # print("pre search results:")
        # print_search_result(results)

        filtered_docs = []
        filtered_metas = []
        filtered_ids = []
        filtered_scores = []

        for doc, meta, doc_id, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['ids'][0],
                results['distances'][0]
        ):
            # 필터링 (cosine distance → similarity = 1 - distance)
            similarity = 1 - distance
            if similarity >= min_score:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_ids.append(doc_id)
                filtered_scores.append(distance)

        post_filtered_results = {
            'documents': [filtered_docs],
            'metadatas': [filtered_metas],
            'ids': [filtered_ids],
            'distances': [filtered_scores]
        }

        print("post search results:")
        print_search_result(post_filtered_results)

        return post_filtered_results