import chromadb
import dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os

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
