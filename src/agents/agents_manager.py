import asyncio
import logging

from src.agents.financial_agent import FinancialAgent
from src.agents.lead_agent import LeadAgent
from src.agents.news_agent import NewsAgent
from src.agents.technical_agent import TechnicalAgent
from src.config.config import Config
from src.data.schemas import LeadAgentReport, AgentReport
from src.utils.llm_client import LLMClient


class AgentsManager:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm_client = LLMClient(config.llm)
        self.lead_agent = LeadAgent(config=config, llm_client=self.llm_client)
        self.news_agent = NewsAgent(config=config, llm_client=self.llm_client)
        self.financial_agent = FinancialAgent(config=config, llm_client=self.llm_client)
        self.technical_agent = TechnicalAgent(config=config, llm_client=self.llm_client)

    async def analyze(self, stocks_data: dict[str, dict]) -> list[LeadAgentReport]:
        logging.info(f"[Phase 2] Running agent analysis on {len(stocks_data)} candidates...")

        semaphore = asyncio.Semaphore(self.config.parallel_workers)
        all_reports: list[LeadAgentReport] = []

        async def analyze_stock(ticker: str):
            async with semaphore:
                return await self._analyze_single_stock(ticker, stocks_data[ticker])

        tasks = [analyze_stock(t) for t in stocks_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, LeadAgentReport):
                all_reports.append(result)
            elif isinstance(result, Exception):
                logging.error(f"Stock analysis failed: {result}")

        return all_reports

    async def _analyze_single_stock(self, ticker: str, cached_data: dict) -> LeadAgentReport:
        """Run all three agents on a single stock using pre-fetched cached data."""
        logging.info(f"  Analyzing {ticker}...")

        company_name = cached_data["company_name"]

        # Build agent input dicts from cache — no fetching here
        news_data = {
            "news": cached_data["news"],
            "company_name": company_name,
        }
        financial_data = {
            "financials": cached_data["financials"],
            "company_name": company_name,
        }
        technical_data = {
            "price_history": cached_data["price_history"],
            "company_name": company_name,
        }

        # Run 3 agents in parallel
        agent_reports: list[AgentReport] = await asyncio.gather(
            self.news_agent.analyze(ticker, news_data),
            self.financial_agent.analyze(ticker, financial_data),
            self.technical_agent.analyze(ticker, technical_data),
        )

        # Lead Agent synthesizes
        lead_report = await self.lead_agent.synthesize(
            ticker=ticker,
            agent_reports=agent_reports,
            stock_data={
                "company_name": company_name,
                "sector": cached_data.get("sector", ""),
            },
        )

        return lead_report
