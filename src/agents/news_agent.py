"""
News Sentiment Agent for EdgeFolio.
Analyzes recent news articles to determine sentiment and impact on stock potential.
"""

import logging

from src.agents.base_agent import BaseAgent
from src.config.config import Config
from src.data.schemas import AgentReport, AgentType
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class NewsAgent(BaseAgent):
    """Analyzes news sentiment for a given stock."""

    agent_type = AgentType.NEWS

    def __init__(self, config: Config, llm_client: LLMClient):
        super().__init__(config, llm_client)

    def build_system_prompt(self) -> str:
        return """You are a senior financial news analyst specializing in stock sentiment analysis.

Your job is to analyze recent news articles about a company and determine:
1. Overall sentiment (bullish, bearish, or neutral)
2. Whether the news suggests the stock price will increase, decrease, or stay flat
3. Key catalysts or risks mentioned in the news
4. How significant and reliable the news sources are

SCORING GUIDELINES:
- 8-10: Overwhelmingly positive news — major catalysts, strong momentum, positive analyst coverage
- 6-7.9: Generally positive — more good news than bad, mild tailwinds
- 4-5.9: Mixed or neutral — balanced news, nothing strongly directional
- 2-3.9: Generally negative — concerning news, headwinds, negative analyst coverage
- 0-1.9: Overwhelmingly negative — major scandals, lawsuits, severe downgrades

Be objective. Don't be swayed by hype. Weigh the credibility of sources.
If there are very few articles, lower your confidence score.

""" + self._build_output_schema()

    def build_user_message(self, ticker: str, data: dict) -> str:
        articles = data.get("news", [])
        company_name = data.get("company_name", ticker)

        if not articles:
            return f"""
Analyze news sentiment for {ticker} ({company_name}).

NO ARTICLES FOUND in the last 30 days.

Given the lack of news, provide a neutral assessment with low confidence.
Note this as "poor" data quality.
"""

        # Format articles for the prompt
        articles_text = ""
        for i, article in enumerate(articles, 1):
            articles_text += f"""
Article {i}:
  Title: {article.get('title', 'N/A')}
  Source: {article.get('source', 'N/A')}
  Date: {article.get('published_at', 'N/A')}
  Summary: {article.get('description', 'N/A')}
"""

        return f"""
Analyze the following news articles for {ticker} ({company_name}).

Number of articles found: {len(articles)}
Time period: Last 30 days

{articles_text}

Based on these articles, provide your sentiment analysis and score.
"""

    async def analyze(self, ticker: str, data: dict) -> AgentReport:
        """Run news sentiment analysis for a stock."""
        logger.info(f"[NewsAgent] Analyzing {ticker}...")

        system_prompt = self.build_system_prompt()
        user_message = self.build_user_message(ticker, data)

        try:
            response = await self.llm_client.call_json(
                system_prompt=system_prompt,
                user_message=user_message,
            )
            report = self._parse_llm_response(response, ticker)
            logger.info(f"[NewsAgent] {ticker} -> score={report.score}, signal={report.signal.value}")
            return report

        except Exception as e:
            logger.error(f"[NewsAgent] Failed for {ticker}: {e}")
            return AgentReport(
                agent_type=self.agent_type,
                ticker=ticker,
                score=self.config._raw.get("scoring", {}).get("missing_data_default_score", 3.0),
                confidence=0.0,
                reasoning=f"Analysis failed: {str(e)}",
                data_quality="poor",
            )