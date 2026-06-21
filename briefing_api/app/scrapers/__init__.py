from app.scrapers.youtube import YouTubeScraper
from app.scrapers.hn import HackerNewsScraper
from app.scrapers.reddit import RedditScraper
from app.scrapers.github import GitHubScraper
from app.scrapers.arxiv import PaperScraper
from app.scrapers.blogs import BlogScraper

SCRAPERS = [
    YouTubeScraper(),
    HackerNewsScraper(),
    RedditScraper(),
    GitHubScraper(),
    PaperScraper(),
    BlogScraper()
]
