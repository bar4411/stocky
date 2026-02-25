import asyncio
import logging
import time
from datetime import datetime

from src.agents.agents_manager import AgentsManager
from src.config.config import Config
from src.data.data_fetcher import DataFetcher
from src.data.schemas import DiscoveryOutput, RiskLevel
from src.data.stocks_prefilter import Prefilter
from src.pipeline.funnel import TournamentFunnel
from infra.base_pipeline import BasePipeline



class StocksRecommenderPipeline(BasePipeline):


    def __init__(self, config_path: str = None):
        self.config = Config.load(config_path)
        self.data_fetcher = None
        self.stocks_data = None
        self.agents_manager = None
        self.top_pick_stocks_lead_agents = None

    def setup(self):
        """Re-initialize non-picklable resources after loading from saved state."""
        if self.agents_manager is not None:
            self.agents_manager = AgentsManager(config=self.config)

    def tear_down(self):
        """Release non-picklable resources before saving to pickle."""
        if self.agents_manager is not None:
            self.agents_manager = None

    def fetch_tickers(self):
        self.data_fetcher = DataFetcher(self.config)
        filtered_stock_tickers = Prefilter(self.config).filter(self.data_fetcher.get_tickers()[:20])
        self.stocks_data = self.data_fetcher.fetch_all_data(filtered_stock_tickers)

    async def run_agents(self):
        self.agents_manager = AgentsManager(config=self.config)
        all_stocks_lead_agents = await self.agents_manager.analyze(self.stocks_data)
        all_stocks_lead_agents.sort(key=lambda r: r.final_score, reverse=True)
        # Filter by minimum confidence
        qualified = [r for r in all_stocks_lead_agents
                     if r.final_score >= self.config.min_confidence]
        self.top_pick_stocks_lead_agents = qualified[: self.config.top_k]

    async def run_funnel(self):
        """Phase 2 (funnel): Tournament funnel — ~54 LLM calls vs ~320."""
        funnel = TournamentFunnel(config=self.config)
        reports = await funnel.run(self.stocks_data)

        # Expose agents_manager so process_output() can read LLM stats unchanged
        self.agents_manager = funnel.agents_manager

        reports.sort(key=lambda r: r.final_score, reverse=True)
        qualified = [r for r in reports if r.final_score >= self.config.min_confidence]
        self.top_pick_stocks_lead_agents = qualified[: self.config.top_k]

    def process_output(self):
        # Convert start_time float to datetime
        run_date = datetime.fromtimestamp(self.config.start_time)
        duration_seconds = time.time() - self.config.start_time
        
        output = DiscoveryOutput(
            run_date=run_date,
            universe=self.config.universe,
            total_stocks_scanned=self.data_fetcher.n_tickers_fetched,
            candidates_analyzed=len(self.stocks_data),
            risk_profile=RiskLevel(self.config.risk_profile),
            top_picks=self.top_pick_stocks_lead_agents,
            pipeline_mode=self.config.pipeline_mode,
            total_llm_calls=self.agents_manager.llm_client.call_count,
            total_duration_seconds=duration_seconds,
        )

        logging.info(f"\n{output.summary()}")
        logging.info(f"LLM Usage: {self.agents_manager.llm_client.get_usage_stats()}")

        return output

    def publish(self):
        pass