import time
import logging
from datetime import datetime
from typing import List
# pyrefly: ignore [missing-import]
import feedparser
from app.scrapers.base import BaseScraper
from app.schemas import ArticleCreate

logger = logging.getLogger("briefing_api.scrapers.youtube")

CHANNELS = {
    "AI Explained": "UCcxMqS48759Jexz5dE6eNkw",
    "Fireship": "UCSJbGtTlrD56v83UCyLy8wA",
    "Two Minute Papers": "UCbfYPyITQ-7t4vibqOhzyFw",
    "Latent Space": "UC2r8gKkSgdH47vskJ51a7pA",
    "Andrej Karpathy": "UCJy2n59YnUyuTL68A2Y_j-g",
    "Theo - t3.gg": "UC-8QAzbLcRCE248qUP5Cgjw",
    "Piyush Garg": "UCGf6G2a9f7S20nSscVntxkw",
    "Chai aur Code": "UC8butISFwT-Wl7EV0hUK0BQ",
    "Telusko": "UC59K-uG2A5ogwIrHw4bmlEg",
    "Y Combinator": "UCcefcg35RqF7GQFAj2pqgUA"
}

class YouTubeScraper(BaseScraper):
    def __init__(self):
        super().__init__("YouTube")

    def scrape(self) -> List[ArticleCreate]:
        articles = []
        for channel_name, channel_id in CHANNELS.items():
            feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
            try:
                logger.info(f"Fetching YouTube feed for {channel_name} ({channel_id})")
                feed = feedparser.parse(feed_url)
                
                if not feed.entries:
                    logger.warning(f"No videos found in feed for channel: {channel_name}")
                    continue

                for entry in feed.entries:
                    # Convert published time struct to datetime
                    if not hasattr(entry, "published_parsed") or not entry.published_parsed:
                        continue
                    
                    pub_time = entry.published_parsed
                    published_dt = datetime(*pub_time[:6])

                    # Check if published in the last 24 hours
                    if not self._is_within_last_24_hours(published_dt):
                        continue

                    video_url = entry.link
                    title = entry.title
                    summary = entry.get("summary", "")
                    if not summary and hasattr(entry, "media_description"):
                        summary = entry.media_description

                    # Fast pre-filtering to save LLM tokens and prevent timeouts
                    if not self._is_relevant(title, summary):
                        continue

                    articles.append(ArticleCreate(
                        title=title,
                        url=video_url,
                        author=channel_name,
                        summary=summary[:1000],  # Truncate summary if too long
                        content=summary,
                        source_type="youtube",
                        published_at=published_dt,
                        article_metadata={
                            "channel_id": channel_id,
                            "video_id": entry.get("yt_videoid", "")
                        }
                    ))
            except Exception as e:
                logger.error(f"Error scraping YouTube channel {channel_name}: {str(e)}", exc_info=True)
        
        logger.info(f"YouTube scraper finished. Found {len(articles)} videos in the last 24 hours.")
        return articles
