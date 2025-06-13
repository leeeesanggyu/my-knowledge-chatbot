import requests
import time

VELGO_GRAPHQL_URL = "https://v2.velog.io/graphql"

POSTS_QUERY = """
query Posts($cursor: ID, $username: String, $temp_only: Boolean, $tag: String, $limit: Int) {
  posts(cursor: $cursor, username: $username, temp_only: $temp_only, tag: $tag, limit: $limit) {
    id
    title
    url_slug
  }
}
"""

POST_DETAIL_QUERY = """
query ($username: String!, $slug: String!) {
  post(username: $username, url_slug: $slug) {
    id
    title
    body
    released_at
    updated_at
  }
}
"""

class VelogCrawler:

    def __init__(self, username, test_mode=False, test_limit=5):
        self.username = username
        self.test_mode = test_mode
        self.test_limit = test_limit

    def fetch_post_list(self):
        all_posts = []
        cursor = None

        while True:
            variables = {
                "username": self.username,
                "cursor": cursor,
                "limit": 100
            }
            payload = {
                "operationName": "Posts",
                "variables": variables,
                "query": POSTS_QUERY
            }

            response = requests.post(VELGO_GRAPHQL_URL, json=payload)
            response.raise_for_status()

            data = response.json()
            posts = data['data']['posts']

            if not posts:
                break

            all_posts.extend(posts)
            cursor = posts[-1]['id']

            print(f"Fetched {len(all_posts)} posts so far...")

            if self.test_mode and len(all_posts) >= self.test_limit:
                all_posts = all_posts[:self.test_limit]
                break

            time.sleep(0.5)

        return all_posts

    def fetch_post_detail(self, slug):
        variables = {
            "username": self.username,
            "slug": slug
        }
        payload = {
            "query": POST_DETAIL_QUERY,
            "variables": variables
        }

        response = requests.post(VELGO_GRAPHQL_URL, json=payload)
        response.raise_for_status()

        data = response.json()
        return data['data']['post']
