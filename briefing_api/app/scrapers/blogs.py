import logging
from datetime import datetime
from typing import List
# pyrefly: ignore [missing-import]
import feedparser
from app.scrapers.base import BaseScraper
from app.schemas import ArticleCreate

logger = logging.getLogger("briefing_api.scrapers.blogs")

BLOG_FEEDS = {
    "OpenAI": "https://openai.com/news/rss.xml",
    "Anthropic": "https://www.anthropic.com/news.rss",
    "Google DeepMind": "https://deepmind.google/blog/rss.xml",
    "Hugging Face": "https://huggingface.co/blog/feed.xml",
    "LangChain": "https://blog.langchain.dev/rss/",
    "LlamaIndex": "https://www.llamaindex.ai/blog/feed.xml",
    "Pinecone": "https://www.pinecone.io/blog/rss.xml",
    "Weaviate": "https://weaviate.io/blog/rss.xml",
    "Redis": "https://redis.com/blog/rss/"
}

# Fallback/secondary feed URLs in case the primary one is rate-limited or fails
BLOG_FEEDS_FALLBACK = {
    "LlamaIndex": "https://medium.com/feed/llamaindex-blog",
    "Redis": "https://redis.com/blog/feed/"
}

class BlogScraper(BaseScraper):
    def __init__(self):
        super().__init__("Blogs")

    def scrape(self) -> List[ArticleCreate]:
        articles = []
        for company_name, feed_url in BLOG_FEEDS.items():
            urls_to_try = [feed_url]
            if company_name in BLOG_FEEDS_FALLBACK:
                urls_to_try.append(BLOG_FEEDS_FALLBACK[company_name])

            success = False
            for url in urls_to_try:
                try:
                    logger.info(f"Fetching RSS blog feed for {company_name} from {url}...")
                    feed = feedparser.parse(url)
                    
                    if not feed.entries and feed.bozo:
                        # Feed has parsing errors, try fallback if available
                        logger.warning(f"Failed to parse feed for {company_name} (Bozo exception). Trying next URL...")
                        continue

                    for entry in feed.entries:
                        if not hasattr(entry, "published_parsed") or not entry.published_parsed:
                            # Try updated parsed if published parsed doesn't exist
                            if hasattr(entry, "updated_parsed") and entry.updated_parsed:
                                pub_time = entry.updated_parsed
                            else:
                                continue
                        else:
                            pub_time = entry.published_parsed

                        published_dt = datetime(*pub_time[:6])

                        # Check if within last 24 hours
                        if not self._is_within_last_24_hours(published_dt):
                            continue

                        title = entry.title
                        link = entry.link
                        
                        # Extract author (default to company name)
                        author = entry.get("author", company_name)
                        if not author:
                            author = company_name

                        # Extract summary / content
                        summary = entry.get("summary", "")
                        if not summary and hasattr(entry, "content"):
                            summary = entry.content[0].value if entry.content else ""
                        
                        articles.append(ArticleCreate(
                            title=title,
                            url=link,
                            author=author,
                            summary=summary[:1000] if summary else f"Recent blog post from {company_name}.",
                            content=summary,
                            source_type="blog",
                            published_at=published_dt,
                            article_metadata={
                                "company": company_name,
                                "feed_url": url
                            }
                        ))
                    
                    success = True
                    break # Managed to scrape successfully, move to next company
                except Exception as e:
                    logger.warning(f"Error scraping {company_name} RSS feed at {url}: {str(e)}")

            if not success:
                logger.error(f"Could not scrape RSS feed for {company_name} using any of the URLs.")

        logger.info(f"Blog scraper finished. Found {len(articles)} blog posts in the last 24 hours.")
        return articles
