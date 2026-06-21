import logging
from abc import ABC, abstractmethod
from typing import List
from datetime import datetime, timedelta
# pyrefly: ignore [missing-import]
import httpx
from app.schemas import ArticleCreate

logger = logging.getLogger("briefing_api.scrapers")
logging.basicConfig(level=logging.INFO)

# Broad interest keywords to perform a fast pre-filter and save local LLM inference time.
INTEREST_KEYWORDS = [
    "llm", "ai", "agent", "mcp", "rag", "vector", "database", "redis", "search", 
    "semantic", "voice", "realtime", "infrastructure", "gpu", "cuda", "onnx", 
    "open source", "geospatial", "openlayers", "deck.gl", "map", "developer tool", 
    "devtool", "startup", "yc", "paper", "launch", "llama", "deepseek", "gemma", 
    "openai", "anthropic", "gemini", "claude", "mistral", "transformer", "backend", 
    "system design", "distributed", "postgres", "sql", "docker", "kubernetes", 
    "api", "concurrency", "performance", "scaling", "webgpu", "frontend"
]

class BaseScraper(ABC):
    def __init__(self, name: str):
        self.name = name
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    @abstractmethod
    def scrape(self) -> List[ArticleCreate]:
        """
        Scrape data source and return a list of ArticleCreate objects.
        """
        pass

    def _get_request(self, url: str, params: dict = None, timeout: int = 15) -> httpx.Response:
        """
        Helper method to execute HTTP GET requests with custom headers.
        """
        with httpx.Client(headers=self.headers, follow_redirects=True) as client:
            response = client.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response

    def _is_within_last_24_hours(self, dt: datetime) -> bool:
        """
        Check if the given datetime is within the last 24 hours.
        """
        now = datetime.utcnow()
        # Handle naive vs aware datetimes
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return now - timedelta(hours=24) <= dt <= now

    def _is_relevant(self, title: str, text: str = "") -> bool:
        """
        Temporarily disabled: returns True to fetch all articles.
        """
        return True
