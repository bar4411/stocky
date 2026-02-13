"""
Lead Agent for EdgeFolio.
Synthesizes reports from all sub-agents into a final stock recommendation.
V1: LLM-based reasoning over agent reports.
V2 (future): Can ask follow-up questions to sub-agents.
"""

import logging

from src.agents.base_agent import BaseAgent
from src.config.config import Config
from src.data.schemas import AgentReport, AgentType, LeadAgentReport, Signal
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class LeadAgent(BaseAgent):
    """Synthesizes sub-agent reports into a final investment recommendation."""

    agent_type = AgentType.LEAD

    def __init__(self, config: Config, llm_client: LLMClient):
        super().__init__(config, llm_client)

    def build_system_prompt(self) -> str:
        risk_profile = self.config.risk_profile

        return f"""You are a senior portfolio strategist synthesizing multiple analyst reports into a final investment recommendation.

You receive reports from three specialized analysts:
1. News Sentiment Analyst — evaluates recent news and market sentiment
2. Financial Analyst — evaluates fundamentals, valuation, and financial health
3. Technical Analyst — evaluates price patterns, momentum, and technical indicators

YOUR ROLE:
- Weigh all three reports holistically (don't just average scores)
- Identify where analysts agree and where they conflict
- Use conflicts as signal: e.g., if news is bullish but technicals are overbought, that's a warning
- Factor in data quality: give less weight to analysts with poor/partial data
- Consider the current risk profile: {risk_profile}

RISK PROFILE CONTEXT:
- Conservative: Prioritize financial health and low volatility. Prefer stable companies with dividends.
- Moderate: Balance growth potential with risk. Standard approach.
- Aggressive: Prioritize upside potential. Tolerate higher risk and volatility.

SCORING GUIDELINES:
- 8-10: Strong buy — multiple analysts align positively, strong conviction
- 6-7.9: Buy — generally positive outlook with manageable risks
- 4-5.9: Hold/Neutral — mixed signals, wait for clarity
- 2-3.9: Underperform — more risks than opportunities identified
- 0-1.9: Avoid — strong negative consensus across analysts

Respond with a JSON object:
{{
    "final_score": <float 0-10>,
    "signal": "<strong_buy|buy|neutral|sell|strong_sell>",
    "confidence": <float 0-1>,
    "investment_thesis": "<2-3 sentence summary of why to buy/avoid this stock>",
    "key_catalysts": ["<catalyst 1>", "<catalyst 2>", "<catalyst 3>"],
    "key_risks": ["<risk 1>", "<risk 2>"],
    "agent_agreement": "<description of how much agents agree/conflict>",
    "score_adjustments": "<explain if/why you adjusted from the raw average>"
}}
"""

    def build_user_message(self, ticker: str, data: dict) -> str:
        agent_reports: list[AgentReport] = data.get("agent_reports", [])
        company_name = data.get("company_name", ticker)
        sector = data.get("sector", "Unknown")

        reports_text = ""
        for report in agent_reports:
            reports_text += f"""
--- {report.agent_type.value.upper()} ANALYST REPORT ---
Score: {report.score}/10
Signal: {report.signal.value}
Confidence: {report.confidence}
Data Quality: {report.data_quality}
Reasoning: {report.reasoning}
Key Factors: {', '.join(report.key_factors)}
Risks: {', '.join(report.risks)}
"""

        weights = self.config.get_agent_weights()
        weights_text = ", ".join(f"{k}: {v:.0%}" for k, v in weights.items())

        return f"""
Synthesize the following analyst reports for {ticker} ({company_name}).
Sector: {sector}
Agent weights (for reference): {weights_text}

{reports_text}

Provide your final synthesized recommendation.
"""

    async def analyze(self, ticker: str, data: dict) -> AgentReport:
        """Not used directly — use synthesize() instead."""
        raise NotImplementedError("Use synthesize() for Lead Agent")

    async def synthesize(
        self, ticker: str, agent_reports: list[AgentReport], stock_data: dict
    ) -> LeadAgentReport:
        """
        Synthesize multiple agent reports into a final recommendation.

        Args:
            ticker: Stock ticker
            agent_reports: List of AgentReport from sub-agents
            stock_data: Additional stock metadata (company name, sector, etc.)
        """
        logger.info(f"[LeadAgent] Synthesizing {ticker} from {len(agent_reports)} agent reports...")

        data = {
            "agent_reports": agent_reports,
            "company_name": stock_data.get("company_name", ticker),
            "sector": stock_data.get("sector", "Unknown"),
        }

        system_prompt = self.build_system_prompt()
        user_message = self.build_user_message(ticker, data)

        try:
            response = await self.llm_client.call_json(
                system_prompt=system_prompt,
                user_message=user_message,
            )

            # Map string signal to enum
            signal_map = {
                "strong_buy": Signal.STRONG_BUY,
                "buy": Signal.BUY,
                "neutral": Signal.NEUTRAL,
                "sell": Signal.SELL,
                "strong_sell": Signal.STRONG_SELL,
            }

            # Extract individual agent scores
            news_score = None
            financial_score = None
            technical_score = None
            for r in agent_reports:
                if r.agent_type == AgentType.NEWS:
                    news_score = r.score
                elif r.agent_type == AgentType.FINANCIAL:
                    financial_score = r.score
                elif r.agent_type == AgentType.TECHNICAL:
                    technical_score = r.score

            report = LeadAgentReport(
                ticker=ticker,
                company_name=stock_data.get("company_name", ticker),
                sector=stock_data.get("sector", ""),
                final_score=float(response.get("final_score", 5.0)),
                signal=signal_map.get(response.get("signal", "neutral"), Signal.NEUTRAL),
                confidence=float(response.get("confidence", 0.5)),
                news_score=news_score,
                financial_score=financial_score,
                technical_score=technical_score,
                investment_thesis=response.get("investment_thesis", ""),
                key_catalysts=response.get("key_catalysts", []),
                key_risks=response.get("key_risks", []),
                agent_agreement=response.get("agent_agreement", ""),
                agent_reports=agent_reports,
            )

            logger.info(
                f"[LeadAgent] {ticker} -> final_score={report.final_score}, "
                f"signal={report.signal.value}"
            )
            return report

        except Exception as e:
            logger.error(f"[LeadAgent] Failed for {ticker}: {e}")
            # Fallback: weighted average
            weights = self.config.get_agent_weights()
            fallback_score = sum(
                r.score * weights.get(r.agent_type.value, 0.33)
                for r in agent_reports
            )
            return LeadAgentReport(
                ticker=ticker,
                company_name=stock_data.get("company_name", ticker),
                final_score=round(fallback_score, 1),
                signal=Signal.NEUTRAL,
                confidence=0.3,
                investment_thesis=f"Fallback score (LLM synthesis failed): {e}",
                agent_reports=agent_reports,
            )