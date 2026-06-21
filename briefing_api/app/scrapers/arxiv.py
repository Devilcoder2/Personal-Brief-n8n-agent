import time
import logging
from datetime import datetime, date
from typing import List
# pyrefly: ignore [missing-import]
import feedparser
from app.scrapers.base import BaseScraper
from app.schemas import ArticleCreate

logger = logging.getLogger("briefing_api.scrapers.arxiv")

class PaperScraper(BaseScraper):
    def __init__(self):
        super().__init__("Papers")

    def scrape(self) -> List[ArticleCreate]:
        articles = []
        
        # 1. Fetch arXiv CS.AI, CS.LG, CS.CL papers
        try:
            logger.info("Fetching papers from arXiv API...")
            # Query for cs.AI (Artificial Intelligence), cs.LG (Machine Learning), cs.CL (Computation and Language)
            arxiv_url = (
                "http://export.arxiv.org/api/query?"
                "search_query=cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL&"
                "sortBy=submittedDate&sortOrder=descending&"
                "max_results=30"
            )
            
            feed = feedparser.parse(arxiv_url)
            
            for entry in feed.entries:
                if not hasattr(entry, "published_parsed") or not entry.published_parsed:
                    continue
                
                pub_time = entry.published_parsed
                published_dt = datetime(*pub_time[:6])

                if not self._is_within_last_24_hours(published_dt):
                    continue

                title = entry.title.replace("\n", " ").strip()
                url = entry.id
                
                # Extract authors
                authors_list = []
                if hasattr(entry, "authors"):
                    authors_list = [a.name for a in entry.authors if hasattr(a, "name")]
                authors = ", ".join(authors_list) if authors_list else "arXiv"

                summary = entry.summary.replace("\n", " ").strip()

                # Fast pre-filtering to save LLM tokens and prevent timeouts
                if not self._is_relevant(title, summary):
                    continue

                articles.append(ArticleCreate(
                    title=title,
                    url=url,
                    author=authors,
                    summary=summary[:1000],
                    content=summary,
                    source_type="arxiv",
                    published_at=published_dt,
                    article_metadata={
                        "arxiv_id": url.split("/abs/")[-1] if "/abs/" in url else "",
                        "arxiv_categories": [t.term for t in entry.tags] if hasattr(entry, "tags") else []
                    }
                ))
        except Exception as e:
            logger.error(f"Error scraping arXiv papers: {str(e)}", exc_info=True)

        # 2. Fetch Hugging Face Daily Papers (most discussed papers)
        try:
            logger.info("Fetching Hugging Face daily papers...")
            # Endpoint returns daily papers
            hf_url = "https://huggingface.co/api/daily_papers"
            response = self._get_request(hf_url)
            papers = response.json()
            
            for p in papers:
                title = p.get("title", "")
                paper_id = p.get("id")
                if not paper_id:
                    continue
                
                # Check if we have publication date, default to today
                published_at_str = p.get("publishedAt")
                published_dt = datetime.utcnow()
                if published_at_str:
                    try:
                        # Clean Z and millisecond notation
                        cleaned_str = published_at_str.split(".")[0].replace("Z", "")
                        published_dt = datetime.strptime(cleaned_str, "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        pass
                
                # Check if published in the last 24 hours
                # Since daily papers is curated, we are more lenient and include the current daily batch
                
                paper_url = f"https://huggingface.co/papers/{paper_id}"
                
                # Check if already scraped from arXiv to avoid duplicate entries
                if any(art.url == paper_url or paper_id in art.url for art in articles):
                    continue

                upvotes = p.get("upvotes", 0)
                
                # Try to extract summary/authors from inner nested fields if they exist
                paper_details = p.get("paper", {}) or {}
                summary = paper_details.get("summary", "").replace("\n", " ").strip()
                if not summary:
                    summary = f"Hugging Face Daily Paper with {upvotes} community discussions."
                
                authors_list = paper_details.get("authors", []) or []
                authors = ", ".join([a.get("name", "") for a in authors_list if a.get("name")])
                if not authors:
                    authors = "Hugging Face Community"

                # Fast pre-filtering to save LLM tokens and prevent timeouts
                if not self._is_relevant(title, summary):
                    continue

                articles.append(ArticleCreate(
                    title=title,
                    url=paper_url,
                    author=authors,
                    summary=summary[:1000],
                    content=summary,
                    source_type="arxiv",  # Keep under arxiv source type for section organization
                    published_at=published_dt,
                    article_metadata={
                        "hf_id": paper_id,
                        "hf_upvotes": upvotes,
                        "is_hf_curated": True
                    }
                ))
        except Exception as e:
            logger.error(f"Error scraping Hugging Face papers: {str(e)}", exc_info=True)

        logger.info(f"Papers scraper finished. Found {len(articles)} papers.")
        return articles
