import logging
from datetime import datetime
from typing import List, Set
from app.scrapers.base import BaseScraper
from app.schemas import ArticleCreate

logger = logging.getLogger("briefing_api.scrapers.hn")

# Broad interest keywords to perform a fast pre-filter and save LLM token costs.
# Any item containing one of these in its title will be forwarded to the LLM scoring layer.
INTEREST_KEYWORDS = [
    "llm", "ai", "agent", "mcp", "rag", "vector", "database", "redis", "search", 
    "semantic", "voice", "realtime", "infrastructure", "gpu", "cuda", "onnx", 
    "open source", "geospatial", "openlayers", "deck.gl", "map", "developer tool", 
    "devtool", "startup", "yc", "paper", "launch", "llama", "deepseek", "gemma", 
    "openai", "anthropic", "gemini", "claude", "mistral", "transformer", "backend", 
    "system design", "distributed", "postgres", "sql", "docker", "kubernetes", 
    "api", "concurrency", "performance", "scaling", "webgpu", "frontend"
]

class HackerNewsScraper(BaseScraper):
    def __init__(self):
        super().__init__("Hacker News")

    def scrape(self) -> List[ArticleCreate]:
        articles = []
        seen_ids: Set[int] = set()
        
        # We check top, best, show, and ask stories
        endpoints = [
            ("top", "https://hacker-news.firebaseio.com/v0/topstories.json"),
            ("best", "https://hacker-news.firebaseio.com/v0/beststories.json"),
            ("show", "https://hacker-news.firebaseio.com/v0/showstories.json"),
            ("ask", "https://hacker-news.firebaseio.com/v0/askstories.json")
        ]

        for category, url in endpoints:
            try:
                logger.info(f"Fetching HN {category} stories list...")
                response = self._get_request(url)
                story_ids = response.json()
                
                # Fetch details for the first 30 stories in this category
                for story_id in story_ids[:30]:
                    if story_id in seen_ids:
                        continue
                    seen_ids.add(story_id)

                    try:
                        item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                        item_response = self._get_request(item_url, timeout=5)
                        item = item_response.json()
                        
                        if not item or item.get("type") != "story":
                            continue

                        # Extract timestamp
                        published_time = item.get("time")
                        if not published_time:
                            continue
                        
                        published_dt = datetime.utcfromtimestamp(published_time)

                        # Check if within last 24 hours
                        if not self._is_within_last_24_hours(published_dt):
                            continue

                        title = item.get("title", "")
                        url_link = item.get("url", f"https://news.ycombinator.com/item?id={story_id}")
                        author = item.get("by", "")
                        score = item.get("score", 0)
                        descendants = item.get("descendants", 0) # Comments count
                        text_content = item.get("text", "") # For Ask HN / Show HN text

                        # Keyword pre-filtering to save local LLM inference time
                        title_lower = title.lower()
                        text_lower = text_content.lower() if text_content else ""
                        is_relevant = any(kw in title_lower or kw in text_lower for kw in INTEREST_KEYWORDS)

                        if not is_relevant:
                            continue

                        articles.append(ArticleCreate(
                            title=title,
                            url=url_link,
                            author=author,
                            summary=text_content[:1000] if text_content else f"HN Story by {author} with {score} points and {descendants} comments.",
                            content=text_content if text_content else "",
                            source_type="hn",
                            published_at=published_dt,
                            article_metadata={
                                "hn_id": story_id,
                                "points": score,
                                "comments_count": descendants,
                                "hn_category": category
                            }
                        ))
                    except Exception as child_e:
                        logger.debug(f"Failed to fetch details for HN item {story_id}: {str(child_e)}")
            except Exception as e:
                logger.error(f"Error scraping HN category {category}: {str(e)}", exc_info=True)

        logger.info(f"Hacker News scraper finished. Found {len(articles)} relevant stories in the last 24 hours.")
        return articles
