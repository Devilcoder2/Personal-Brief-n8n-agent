import logging
import re
from datetime import datetime, timedelta
from typing import List
# pyrefly: ignore [missing-import]
from bs4 import BeautifulSoup   
from app.scrapers.base import BaseScraper
from app.schemas import ArticleCreate

logger = logging.getLogger("briefing_api.scrapers.github")

class GitHubScraper(BaseScraper):
    def __init__(self):
        super().__init__("GitHub")

    def scrape(self) -> List[ArticleCreate]:
        articles = []
        
        # 1. Scrape standard GitHub trending (overall)
        try:
            logger.info("Scraping GitHub daily trending page...")
            response = self._get_request("https://github.com/trending")
            html = response.text
            articles.extend(self._parse_trending_html(html))
        except Exception as e:
            logger.error(f"Error scraping GitHub trending HTML: {str(e)}", exc_info=True)

        # 2. Query GitHub Search API for MCP and AI repositories (as a robust API-based fallback/supplement)
        try:
            logger.info("Querying GitHub Search API for new/trending MCP repositories...")
            # Query for repositories created/updated in the last 7 days containing 'model-context-protocol' or 'mcp'
            past_week = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
            url = "https://api.github.com/search/repositories"
            params = {
                "q": f"(model-context-protocol OR mcp OR llm-agent OR local-llm) pushed:>{past_week}",
                "sort": "stars",
                "order": "desc",
                "per_page": 20
            }
            
            response = self._get_request(url, params=params)
            search_data = response.json()
            items = search_data.get("items", [])
            
            for item in items:
                repo_name = item.get("full_name")
                repo_url = item.get("html_url")
                description = item.get("description") or ""
                stars = item.get("stargazers_count", 0)
                forks = item.get("forks_count", 0)
                lang = item.get("language") or "N/A"
                owner = item.get("owner", {}).get("login", "")
                created_at_str = item.get("created_at")
                
                published_dt = datetime.utcnow()
                if created_at_str:
                    try:
                        published_dt = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        pass
                
                # Check if we already have it in the list (from the trending page)
                if any(art.url == repo_url for art in articles):
                    continue

                summary = f"GitHub repository: {repo_name} ({lang}). Stars: {stars}, Forks: {forks}. Description: {description}"
                
                articles.append(ArticleCreate(
                    title=f"GitHub Repo: {repo_name}",
                    url=repo_url,
                    author=owner,
                    summary=summary[:1000],
                    content=description,
                    source_type="github",
                    published_at=published_dt,
                    article_metadata={
                        "stars": stars,
                        "forks": forks,
                        "language": lang,
                        "repo_name": repo_name,
                        "is_mcp": "mcp" in repo_name.lower() or "mcp" in description.lower()
                    }
                ))
        except Exception as e:
            logger.error(f"Error calling GitHub Search API: {str(e)}", exc_info=True)

        logger.info(f"GitHub scraper finished. Found {len(articles)} repositories.")
        return articles

    def _parse_trending_html(self, html: str) -> List[ArticleCreate]:
        repos = []
        soup = BeautifulSoup(html, "lxml")
        
        # GitHub trending items are in <article class="Box-row">
        repo_rows = soup.find_all("article", class_="Box-row")
        for row in repo_rows:
            try:
                # Get repo path (e.g. /owner/repo)
                title_a = row.find("h2", class_="h3").find("a")
                repo_path = title_a.get("href", "").strip()
                if repo_path.startswith("/"):
                    repo_path = repo_path[1:]
                
                repo_url = f"https://github.com/{repo_path}"
                owner = repo_path.split("/")[0] if "/" in repo_path else ""

                # Description
                desc_p = row.find("p", class_="col-9")
                description = desc_p.text.strip() if desc_p else ""

                # Programming Language
                lang_span = row.find("span", itemprop="programmingLanguage")
                lang = lang_span.text.strip() if lang_span else "N/A"

                # Stars & Forks (contained in anchor tags relative to repo path)
                stars_a = row.find("a", href=re.compile(f"^/{repo_path}/stargazers"))
                stars_text = stars_a.text.strip().replace(",", "") if stars_a else "0"
                stars = int(stars_text) if stars_text.isdigit() else 0

                forks_a = row.find("a", href=re.compile(f"^/{repo_path}/forks"))
                forks_text = forks_a.text.strip().replace(",", "") if forks_a else "0"
                forks = int(forks_text) if forks_text.isdigit() else 0

                # New Stars Today
                today_span = row.find("span", class_="float-sm-right")
                today_text = today_span.text.strip() if today_span else ""
                
                summary = f"GitHub Trending Repo: {repo_path} ({lang}). Stars: {stars} (+{today_text}), Forks: {forks}. Description: {description}"
                
                repos.append(ArticleCreate(
                    title=f"GitHub Repo: {repo_path}",
                    url=repo_url,
                    author=owner,
                    summary=summary[:1000],
                    content=description,
                    source_type="github",
                    published_at=datetime.utcnow(),  # Trending is current
                    article_metadata={
                        "stars": stars,
                        "forks": forks,
                        "language": lang,
                        "repo_name": repo_path,
                        "stars_today": today_text,
                        "is_mcp": "mcp" in repo_path.lower() or "mcp" in description.lower()
                    }
                ))
            except Exception as row_e:
                logger.debug(f"Failed to parse a row in GitHub Trending HTML: {str(row_e)}")
                
        return repos
