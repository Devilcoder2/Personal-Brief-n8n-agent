import logging
from datetime import datetime, date, timedelta
import httpx
from sqlalchemy.orm import Session
from app.config import settings
from app.models import Article, Briefing

logger = logging.getLogger("briefing_api.services.insights")

class InsightsService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_LLM_MODEL

    def generate_daily_briefing(self, db: Session, target_date: date) -> tuple[Briefing, str]:
        """
        Gathers YouTube videos from the last 24 hours, filters and groups them,
        calls Ollama to generate personalized insights, structures the markdown and HTML,
        and saves the briefing to the database.
        """
        start_time = datetime.combine(target_date, datetime.min.time()) - timedelta(hours=2)
        end_time = datetime.combine(target_date, datetime.max.time()) + timedelta(hours=2)

        # Get all non-duplicate ranked videos from today, sorted by published_at DESC (newest first)
        articles = (
            db.query(Article)
            .filter(
                Article.is_duplicate == False,
                Article.overall_score.is_not(None),
                Article.source_type == "youtube",
                Article.created_at >= start_time,
                Article.created_at <= end_time
            )
            .order_by(Article.published_at.desc())
            .all()
        )

        if not articles:
            logger.warning(f"No videos found to generate briefing for date {target_date}")
            empty_content = (
                f"# Daily Developer Video Briefing\n"
                f"*Date: {target_date}*\n\n"
                f"No new developer videos were captured for today."
            )
            empty_html = self._compile_html(target_date, [], [], "", [])
            
            # Save empty briefing
            existing_briefing = db.query(Briefing).filter(Briefing.date == target_date).first()
            if existing_briefing:
                existing_briefing.content = empty_content
                db.commit()
                return existing_briefing, empty_html
            else:
                briefing = Briefing(date=target_date, content=empty_content)
                db.add(briefing)
                db.commit()
                return briefing, empty_html

        # Must Watch: Top 2 videos
        must_watch = articles[:2]
        must_watch_ids = {a.id for a in must_watch}

        # Rest of the videos
        other_videos = [a for a in articles if a.id not in must_watch_ids][:6]

        # Call LLM to generate Personalized Insights based on top video descriptions
        insights_text = self._generate_personalized_insights_llm(must_watch, other_videos)

        # Assemble Markdown Briefing
        markdown = self._compile_markdown(
            target_date=target_date,
            must_watch=must_watch,
            other_videos=other_videos,
            insights=insights_text,
            all_articles=articles
        )

        # Parse insights markdown to HTML
        insights_html = self._markdown_to_html(insights_text)

        # Assemble HTML Briefing
        html = self._compile_html(
            target_date=target_date,
            must_watch=must_watch,
            other_videos=other_videos,
            insights_html=insights_html,
            all_articles=articles
        )

        # Save to DB (override if briefing already exists for this date)
        existing_briefing = db.query(Briefing).filter(Briefing.date == target_date).first()
        if existing_briefing:
            existing_briefing.content = markdown
            db.commit()
            return existing_briefing, html
        else:
            briefing = Briefing(date=target_date, content=markdown)
            db.add(briefing)
            db.commit()
            return briefing, html

    def _generate_personalized_insights_llm(self, must_watch: list[Article], other_videos: list[Article]) -> str:
        """
        Invokes Ollama to synthesize the daily video summaries and write custom developer notes.
        """
        context_lines = []
        context_lines.append("### TOP MUST WATCH VIDEOS")
        for i, a in enumerate(must_watch, 1):
            context_lines.append(f"{i}. {a.title} by {a.author}\n   Summary: {a.summary or 'N/A'}")
        
        if other_videos:
            context_lines.append("\n### OTHER RELEVANT RELEASES")
            for a in other_videos:
                context_lines.append(f"- {a.title} by {a.author}\n  Summary: {a.summary or 'N/A'}")

        top_developments_context = "\n".join(context_lines)

        system_prompt = (
            "You are a Staff AI Architect and Tech Analyst. Your job is to synthesize today's major video releases "
            "from top developer channels and provide actionable insights for an experienced software developer interested in "
            "Backend Engineering, System Design, LLMs, AI Agents, MCP (Model Context Protocol), RAG, and Developer Tools.\n\n"
            "Format your response as a professional analysis with the following headings:\n"
            "#### Key Takeaways & Trends\n"
            "<Explain the core trends or announcements shown in today's videos>\n\n"
            "#### Impact on AI & Backend Engineers\n"
            "<How these video concepts/launches affect design choices, development workflows, or architectural patterns>\n\n"
            "#### Watch Recommendation\n"
            "<Recommend specifically which 1-2 videos are worth watching immediately and which ones to skip. Be direct.>\n\n"
            "Keep your output clean, brief, and in markdown format."
        )

        user_prompt = (
            f"Here are the details of developer videos uploaded in the last 24 hours:\n\n"
            f"{top_developments_context}\n\n"
            f"Please generate the personalized insights analysis."
        )

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        }

        try:
            # We only generate insights if the user has LLM enabled or has running models
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                return result.get("response", "").strip()
        except Exception as e:
            logger.error(f"Error compiling LLM insights: {str(e)}")
            return (
                "#### Key Takeaways & Trends\n"
                "New technical tutorials and tooling reviews were published today by leading developer channels.\n\n"
                "#### Impact on AI & Backend Engineers\n"
                "New APIs and updates in agent capabilities make standard implementations faster to ship.\n\n"
                "#### Watch Recommendation\n"
                "Inspect the video catalog below for channel uploads matching your active research areas."
            )

    def _markdown_to_html(self, md_text: str) -> str:
        """
        Robustly converts the LLM's simple markdown structure (headers, bold text, lists) into HTML.
        """
        if not md_text:
            return ""
        import re
        
        html = md_text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Parse bold (**text**)
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        
        # Parse italic (*text*)
        html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
        
        # Parse inline code (`code`)
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
        
        # Parse bullet lists
        lines = html.split('\n')
        in_list = False
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('- ') or stripped.startswith('* '):
                content = stripped[2:]
                if not in_list:
                    new_lines.append('<ul class="insights-list">')
                    in_list = True
                new_lines.append(f'<li class="insights-list-item">{content}</li>')
            else:
                if in_list:
                    new_lines.append('</ul>')
                    in_list = False
                new_lines.append(line)
        if in_list:
            new_lines.append('</ul>')
        html = '\n'.join(new_lines)
        
        # Parse headers
        html = re.sub(r'^#### (.*)$', r'<h4 class="insights-sub-header">\1</h4>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.*)$', r'<h3 class="insights-sub-header">\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*)$', r'<h2 class="insights-header">\1</h2>', html, flags=re.MULTILINE)
        
        # Parse paragraphs
        paragraphs = html.split('\n\n')
        for i, p in enumerate(paragraphs):
            p_stripped = p.strip()
            # Wrap standard text sections in paragraph blocks, but avoid wrapping HTML containers/lists/headers
            if p_stripped and not p_stripped.startswith('<h') and not p_stripped.startswith('<u') and not p_stripped.startswith('<l') and not p_stripped.startswith('</u') and not p_stripped.startswith('<div') and not p_stripped.startswith('</div'):
                paragraphs[i] = f'<p class="insights-p">{p_stripped}</p>'
        
        html = '\n'.join(paragraphs)
        return html

    def _compile_html(self, target_date: date, must_watch: list[Article], other_videos: list[Article], 
                      insights_html: str, all_articles: list[Article]) -> str:
        """
        Assembles a premium, highly-styled, responsive light-mode HTML template for the email briefing.
        """
        formatted_date = target_date.strftime('%B %d, %Y')
        
        # Build Executive Synthesis card
        if insights_html:
            insights_html_block = f"""
            <div class="insights-card">
              <div class="insights-title">⭐ Executive AI Synthesis</div>
              {insights_html}
            </div>
            """
        else:
            insights_html_block = ""

        # Build Must Watch list
        must_watch_html_list = []
        if must_watch:
            for a in must_watch:
                summary = a.summary or "No summary available."
                must_watch_html_list.append(f"""
                <div class="must-watch-card">
                  <h3><a href="{a.url}" target="_blank">{a.title}</a></h3>
                  <div class="meta-row">
                    <span class="channel-badge">{a.author}</span>
                    <a class="btn-watch" href="{a.url}" target="_blank">Watch Video ➔</a>
                  </div>
                  <blockquote>{summary}</blockquote>
                </div>
                """)
            must_watch_html = "\n".join(must_watch_html_list)
        else:
            must_watch_html = '<p style="color:#64748b; font-style:italic; margin-bottom:20px;">No high-priority videos cataloged today.</p>'

        # Build Other Videos list
        other_videos_html_list = []
        if other_videos:
            for a in other_videos:
                desc = a.summary or "No description available"
                other_videos_html_list.append(f"""
                <li class="video-list-item">
                  <a class="video-title-link" href="{a.url}" target="_blank">{a.title}</a>
                  <span class="channel-badge-small">{a.author}</span>
                  <p class="desc">{desc}</p>
                </li>
                """)
            other_videos_html = "\n".join(other_videos_html_list)
        else:
            other_videos_html = '<p style="color:#64748b; font-style:italic; padding:10px 0;">No other video uploads captured today.</p>'

        # Build Reference Catalogue rows
        reference_rows_list = []
        for a in all_articles:
            reference_rows_list.append(f"""
            <tr>
              <td><code>{a.author}</code></td>
              <td>{a.title}</td>
              <td><a class="ref-link" href="{a.url}" target="_blank">YouTube Link</a></td>
            </tr>
            """)
        reference_rows = "\n".join(reference_rows_list) if reference_rows_list else '<tr><td colspan="3" style="text-align:center; color:#64748b;">No reference videos available today.</td></tr>'

        # Premium HTML Template
        email_template = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Daily Developer Video Briefing</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      color: #1e293b;
      background-color: #f8fafc;
      margin: 0;
      padding: 40px 10px;
      -webkit-font-smoothing: antialiased;
    }}
    .wrapper {{
      max-width: 640px;
      margin: 0 auto;
      background-color: #ffffff;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
      border: 1px solid #e2e8f0;
    }}
    .top-gradient-bar {{
      height: 6px;
      background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    }}
    .header {{
      padding: 40px 30px 25px 30px;
      text-align: center;
      background-color: #ffffff;
    }}
    .header h1 {{
      margin: 0;
      font-size: 24px;
      font-weight: 800;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: #0f172a;
    }}
    .header p {{
      margin: 8px 0 0 0;
      font-size: 13.5px;
      color: #64748b;
      font-weight: 500;
    }}
    .date-badge {{
      display: inline-block;
      margin-top: 14px;
      background-color: #e0e7ff;
      color: #4338ca;
      font-size: 12px;
      font-weight: 700;
      padding: 4px 14px;
      border-radius: 100px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .content {{
      padding: 10px 30px 35px 30px;
    }}
    .section-header {{
      color: #0f172a;
      font-size: 16px;
      font-weight: 800;
      margin-top: 35px;
      margin-bottom: 20px;
      padding-bottom: 8px;
      border-bottom: 2px solid #f1f5f9;
      text-transform: uppercase;
      letter-spacing: 1px;
    }}
    .insights-card {{
      background-color: #faf5ff;
      border-left: 4px solid #a855f7;
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 30px;
    }}
    .insights-title {{
      font-size: 12px;
      font-weight: 800;
      color: #7e22ce;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 16px;
    }}
    .insights-p {{
      font-size: 14px;
      color: #3b0764;
      margin-bottom: 14px;
      line-height: 1.6;
    }}
    .insights-sub-header {{
      font-size: 13.5px;
      font-weight: 700;
      color: #581c87;
      margin-top: 20px;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .insights-list {{
      margin: 8px 0 16px 0;
      padding-left: 20px;
    }}
    .insights-list-item {{
      font-size: 14px;
      color: #3b0764;
      margin-bottom: 6px;
      line-height: 1.55;
    }}
    .must-watch-card {{
      background-color: #ffffff;
      border: 1px solid #e2e8f0;
      border-left: 4px solid #f59e0b;
      border-radius: 10px;
      padding: 24px;
      margin-bottom: 20px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
    }}
    .must-watch-card h3 {{
      margin: 0 0 12px 0;
      font-size: 17px;
      font-weight: 700;
      line-height: 1.45;
    }}
    .must-watch-card h3 a {{
      color: #0f172a;
      text-decoration: none;
    }}
    .must-watch-card h3 a:hover {{
      color: #4f46e5;
    }}
    .meta-row {{
      margin-bottom: 16px;
    }}
    .channel-badge {{
      display: inline-block;
      background-color: #fef3c7;
      color: #d97706;
      font-size: 11px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-right: 12px;
    }}
    .btn-watch {{
      display: inline-block;
      color: #4f46e5;
      text-decoration: none;
      font-weight: 700;
      font-size: 13px;
    }}
    .btn-watch:hover {{
      text-decoration: underline;
    }}
    .must-watch-card blockquote {{
      margin: 16px 0 0 0;
      padding-left: 14px;
      border-left: 2px solid #e2e8f0;
      font-style: normal;
      font-size: 13.5px;
      color: #475569;
      line-height: 1.6;
    }}
    .video-list {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    .video-list-item {{
      border-bottom: 1px solid #f1f5f9;
      padding: 16px 0;
    }}
    .video-list-item:last-child {{
      border-bottom: none;
      padding-bottom: 0;
    }}
    .video-list-item:first-child {{
      padding-top: 0;
    }}
    .video-title-link {{
      color: #0f172a;
      font-weight: 600;
      font-size: 15px;
      text-decoration: none;
      line-height: 1.4;
    }}
    .video-title-link:hover {{
      color: #4f46e5;
    }}
    .channel-badge-small {{
      display: inline-block;
      background-color: #f1f5f9;
      color: #475569;
      font-size: 10px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-left: 8px;
      vertical-align: middle;
    }}
    .video-list-item .desc {{
      margin: 6px 0 0 0;
      font-size: 13.5px;
      color: #64748b;
      line-height: 1.5;
    }}
    code {{
      font-family: Menlo, Consolas, Monaco, monospace;
      background-color: #f1f5f9;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 12px;
      color: #0f172a;
    }}
    .ref-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 15px;
      font-size: 12px;
    }}
    .ref-table th, .ref-table td {{
      border: 1px solid #e2e8f0;
      padding: 10px 12px;
      text-align: left;
    }}
    .ref-table th {{
      background-color: #f8fafc;
      color: #475569;
      font-weight: 700;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.5px;
    }}
    .ref-table tr:nth-child(even) {{
      background-color: #f8fafc;
    }}
    .ref-link {{
      color: #4f46e5;
      text-decoration: none;
      font-weight: 600;
    }}
    .ref-link:hover {{
      text-decoration: underline;
    }}
    .footer {{
      background-color: #f8fafc;
      color: #64748b;
      padding: 30px;
      text-align: center;
      font-size: 12px;
      border-top: 1px solid #e2e8f0;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="top-gradient-bar"></div>
    <div class="header">
      <h1>Daily Developer Briefing</h1>
      <p>Self-Hosted AI Tech Analyst & Executive Video Summary</p>
      <div class="date-badge">{formatted_date}</div>
    </div>
    <div class="content">
      {insights_html_block}
      
      <div class="section-header">
        🔥 Must Watch Videos
      </div>
      {must_watch_html}
      
      <div class="section-header">
        🎥 More Video Releases
      </div>
      <ul class="video-list">
        {other_videos_html}
      </ul>
      
      <div class="section-header">
        🔗 Reference Catalogue
      </div>
      <table class="ref-table">
        <thead>
          <tr>
            <th>Channel</th>
            <th>Video Title</th>
            <th>Link</th>
          </tr>
        </thead>
        <tbody>
          {reference_rows}
        </tbody>
      </table>
    </div>
    <div class="footer">
      Generated automatically via Docker, pgvector, Ollama & n8n.<br>
      Zero recurring SaaS charges. Keep building.
    </div>
  </div>
</body>
</html>"""
        return email_template

    def _compile_markdown(self, target_date: date, must_watch: list[Article], other_videos: list[Article], 
                          insights: str, all_articles: list[Article]) -> str:
        """
        Combines sections into a single standard Markdown document.
        """
        md = []
        md.append(f"# Daily Developer Video Briefing")
        md.append(f"**Date:** {target_date.strftime('%B %d, %Y')}")
        md.append("")
        md.append("---")
        md.append("")

        # Section 1: Must Watch
        md.append("## 🔥 Must Watch")
        if must_watch:
            for a in must_watch:
                md.append(f"### [{a.title}]({a.url})")
                md.append(f"**Channel:** `{a.author}` | **Watch Link:** [YouTube URL]({a.url})")
                md.append(f"> {a.summary or 'No summary available.'}")
                md.append("")
        else:
            md.append("*No high-priority videos cataloged today.*")
            md.append("")

        # Section 2: Other Videos
        md.append("## 🎥 More Video Releases")
        if other_videos:
            for a in other_videos:
                md.append(f"- **[{a.title}]({a.url})** - `{a.author}`")
                md.append(f"  *Description: {a.summary or 'No description available'}*")
            md.append("")
        else:
            md.append("*No other video uploads captured today.*")
            md.append("")

        # Section 3: Personalized Insights
        md.append("## ⭐ Analyst Insights")
        md.append(insights)
        md.append("")

        # Section 4: References
        md.append("## 🔗 Sources & Videos Reference")
        md.append("| Channel | Video Title | Link |")
        md.append("| --- | --- | --- |")
        for a in all_articles:
            md.append(f"| `{a.author}` | {a.title[:60]}... | [YouTube Link]({a.url}) |")
        md.append("")

        return "\n".join(md)

insights_service = InsightsService()

