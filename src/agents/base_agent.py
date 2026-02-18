"""
Base Agent for EdgeFolio.
All agents (news, financial, technical, lead) inherit from this.
Defines the shared interface and common utilities.
"""

import logging
from abc import ABC, abstractmethod

from src.config.config import Config
from src.data.schemas import AgentReport, AgentType
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all EdgeFolio agents.

    V1 (LLM mode): Agent receives pre-fetched data and calls LLM once.
    V2 (Agentic mode): Agent has tools and runs in a reasoning loop.
    """

    agent_type: AgentType

    def __init__(self, config: Config, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client

    @abstractmethod
    async def analyze(self, ticker: str, data: dict) -> AgentReport:
        """
        Analyze a stock and return a structured report.

        Args:
            ticker: Stock ticker symbol
            data: Pre-fetched data relevant to this agent

        Returns:
            AgentReport with score, signal, reasoning, etc.
        """
        pass

    @abstractmethod
    def build_system_prompt(self) -> str:
        """Build the system prompt defining this agent's role and expertise."""
        pass

    @abstractmethod
    def build_user_message(self, ticker: str, data: dict) -> str:
        """Build the user message with the data for analysis."""
        pass

    def _parse_llm_response(self, response: dict, ticker: str) -> AgentReport:
        """Parse the LLM JSON response into an AgentReport."""
        from src.data.schemas import Signal

        # Map string signals to enum
        signal_map = {
            "strong_buy": Signal.STRONG_BUY,
            "buy": Signal.BUY,
            "neutral": Signal.NEUTRAL,
            "sell": Signal.SELL,
            "strong_sell": Signal.STRONG_SELL,
        }

        score = float(response.get("score", 5.0))
        score = max(0.0, min(10.0, score))  # Clamp to 0-10

        return AgentReport(
            agent_type=self.agent_type,
            ticker=ticker,
            score=score,
            signal=signal_map.get(response.get("signal", "neutral"), Signal.NEUTRAL),
            confidence=float(response.get("confidence", 0.5)),
            reasoning=response.get("reasoning", "No reasoning provided"),
            key_factors=response.get("key_factors", []),
            risks=response.get("risks", []),
            data_quality=response.get("data_quality", "good"),
            raw_data_summary=response.get("data_summary", {}),
        )

    def _build_output_schema(self) -> str:
        """JSON schema instruction appended to all agent prompts."""
        return """
Respond with a JSON object in this exact format:
{
    "score": <float 0-10>,
    "signal": "<strong_buy|buy|neutral|sell|strong_sell>",
    "confidence": <float 0-1>,
    "reasoning": "<detailed explanation, 2-4 sentences>",
    "key_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
    "risks": ["<risk 1>", "<risk 2>"],
    "data_quality": "<good|partial|poor>",
    "data_summary": {<key metrics you used in your analysis>}
}
"""