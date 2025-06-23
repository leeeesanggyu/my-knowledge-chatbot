import os
import dotenv
from post_crawler import VelogCrawler
from vector_store import VelogVectorStore
import time

dotenv.load_dotenv()

VELOG_USERNAME = os.getenv("VELOG_USERNAME")


def main():
    username = VELOG_USERNAME
    TEST_MODE = False

    crawler = VelogCrawler(username, test_mode=TEST_MODE, test_limit=3)
    vector_store = VelogVectorStore()

    posts = crawler.fetch_post_list()
    print(f"{len(posts)} 개의 포스트 수집 완료!\n")

    for idx, post in enumerate(posts, start=1):
        slug = post['url_slug']
        print(f"[{idx}/{len(posts)}] {slug} 상세 조회 중...")
        detail = crawler.fetch_post_detail(slug)

        doc_id = detail['id']
        content = detail['body'] or ""
        metadata = {
            "title": detail['title'],
            "released_at": detail['released_at'],
            "url_slug": slug
        }

        vector_store.add_post(doc_id, content, metadata)
        time.sleep(0.5)

    print("\n✅ 전체 벡터 DB 저장 완료!")


if __name__ == "__main__":
    main()
