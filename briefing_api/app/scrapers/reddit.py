import logging
from datetime import datetime
from typing import List
from app.scrapers.base import BaseScraper
from app.schemas import ArticleCreate

logger = logging.getLogger("briefing_api.scrapers.reddit")

SUBREDDITS = [
    "LocalLLaMA",
    "MachineLearning",
    "OpenAI",
    "Artificial",
    "Singularity"
]

class RedditScraper(BaseScraper):
    def __init__(self):
        super().__init__("Reddit")
        # Use base browser headers but specify Accept type
        self.headers["Accept"] = "application/json,application/xml"

    def scrape(self) -> List[ArticleCreate]:
        articles = []
        for sub in SUBREDDITS:
            parsed_successfully = False
            
            # 1. Attempt JSON parsing
            try:
                url = f"https://www.reddit.com/r/{sub}/hot.json"
                params = {"limit": 25}
                logger.info(f"Fetching Reddit posts (JSON) for r/{sub}...")
                response = self._get_request(url, params=params)
                data = response.json()
                
                children = data.get("data", {}).get("children", [])
                for child in children:
                    post_data = child.get("data", {})
                    if not post_data or post_data.get("stickied"):
                        continue
                    
                    created_utc = post_data.get("created_utc")
                    if not created_utc:
                        continue
                    
                    published_dt = datetime.utcfromtimestamp(created_utc)
                    if not self._is_within_last_24_hours(published_dt):
                        continue

                    title = post_data.get("title", "")
                    permalink = post_data.get("permalink", "")
                    post_url = post_data.get("url", f"https://www.reddit.com{permalink}")
                    author = post_data.get("author", "")
                    ups = post_data.get("ups", 0)
                    num_comments = post_data.get("num_comments", 0)
                    selftext = post_data.get("selftext", "")

                    # Check relevance
                    if not self._is_relevant(title, selftext):
                        continue

                    final_url = post_url
                    if not final_url.startswith("http"):
                        final_url = f"https://www.reddit.com{permalink}"

                    articles.append(ArticleCreate(
                        title=title,
                        url=final_url,
                        author=f"r/{sub} - u/{author}",
                        summary=selftext[:1000] if selftext else f"Reddit discussion in r/{sub} with {ups} upvotes.",
                        content=selftext if selftext else "",
                        source_type="reddit",
                        published_at=published_dt,
                        article_metadata={
                            "subreddit": sub,
                            "upvotes": ups,
                            "comments_count": num_comments,
                            "permalink": f"https://www.reddit.com{permalink}",
                            "is_rss_fallback": False
                        }
                    ))
                parsed_successfully = True
            except Exception as json_e:
                logger.warning(f"Reddit JSON feed for r/{sub} failed: {str(json_e)}. Trying RSS fallback...")
            
            # 2. RSS Fallback if JSON failed/blocked
            if not parsed_successfully:
                try:
                    rss_url = f"https://www.reddit.com/r/{sub}/hot/.rss"
                    logger.info(f"Fetching Reddit RSS feed for r/{sub} from {rss_url}...")
                    import feedparser
                    feed = feedparser.parse(rss_url)
                    
                    for entry in feed.entries:
                        if not hasattr(entry, "published_parsed") or not entry.published_parsed:
                            continue
                        
                        pub_time = entry.published_parsed
                        published_dt = datetime(*pub_time[:6])
                        
                        if not self._is_within_last_24_hours(published_dt):
                            continue
                        
                        title = entry.title
                        link = entry.link
                        author = entry.get("author", f"r/{sub}")
                        content = entry.get("summary", "") # RSS summary contains HTML content description
                        
                        # relevance check
                        if not self._is_relevant(title, content):
                            continue
                            
                        articles.append(ArticleCreate(
                            title=title,
                            url=link,
                            author=f"r/{sub} - {author}",
                            summary=content[:1000] if content else f"Reddit RSS discussion in r/{sub}.",
                            content=content,
                            source_type="reddit",
                            published_at=published_dt,
                            article_metadata={
                                "subreddit": sub,
                                "is_rss_fallback": True
                            }
                        ))
                except Exception as rss_e:
                    logger.error(f"Error scraping Reddit r/{sub} RSS fallback: {str(rss_e)}", exc_info=True)
                    
        logger.info(f"Reddit scraper finished. Found {len(articles)} relevant posts.")
        return articles
