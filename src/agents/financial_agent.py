"""
Financial Analysis Agent for EdgeFolio.
Analyzes fundamental data: financial ratios, earnings, revenue, balance sheet health.
"""

import logging

from src.agents.base_agent import BaseAgent
from src.config.config import Config
from src.data.schemas import AgentReport, AgentType
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class FinancialAgent(BaseAgent):
    """Analyzes financial fundamentals for a given stock."""

    agent_type = AgentType.FINANCIAL

    def __init__(self, config: Config, llm_client: LLMClient):
        super().__init__(config, llm_client)

    def build_system_prompt(self) -> str:
        return """You are a senior financial analyst specializing in fundamental stock analysis.

Your job is to evaluate a company's financial health and growth potential based on:
1. Valuation: P/E ratio, PEG, Price-to-Book — is the stock fairly valued, overvalued, or undervalued?
2. Growth: Revenue growth, earnings growth — is the company growing?
3. Profitability: Profit margins, ROE, ROA — is the company efficiently profitable?
4. Balance sheet: Debt-to-equity, current ratio, free cash flow — is the company financially healthy?
5. Analyst consensus: What do Wall Street analysts think?

SCORING GUIDELINES:
- 8-10: Exceptional fundamentals — strong growth, low debt, undervalued, high profitability
- 6-7.9: Good fundamentals — solid financials with some room for improvement
- 4-5.9: Average — nothing stands out positively or negatively
- 2-3.9: Below average — concerning metrics like high debt, declining revenue, or overvaluation
- 0-1.9: Poor fundamentals — severe financial distress, negative cash flow, extreme overvaluation

Compare metrics to typical ranges for the company's sector.
A P/E of 30 is high for utilities but normal for tech.

If key metrics are missing, note that in data_quality and lower confidence.

""" + self._build_output_schema()

    def build_user_message(self, ticker: str, data: dict) -> str:
        financials = data.get("financials", {})
        ratios = financials.get("ratios", {})
        income = financials.get("income_statement", {})
        company_name = data.get("company_name", ticker)
        sector = financials.get("sector", "Unknown")
        industry = financials.get("industry", "Unknown")

        # Format ratios nicely
        ratios_text = ""
        for key, value in ratios.items():
            if value is not None:
                if isinstance(value, float):
                    ratios_text += f"  {key}: {value:.4f}\n"
                else:
                    ratios_text += f"  {key}: {value}\n"

        # Format income statement
        income_text = ""
        for key, value in income.items():
            if value:
                income_text += f"  {key}:\n"
                for date, amount in value.items():
                    income_text += f"    {date}: {amount:,.0f}\n"

        summary = financials.get("company_summary", "No summary available")

        return f"""
Analyze the financial fundamentals of {ticker} ({company_name}).
Sector: {sector} | Industry: {industry}

Company Summary: {summary}

KEY FINANCIAL RATIOS:
{ratios_text if ratios_text else "  No ratio data available"}

INCOME STATEMENT (Quarterly):
{income_text if income_text else "  No income data available"}

Based on these financials, provide your fundamental analysis and score.
Consider the sector context when evaluating ratios.
"""

    async def analyze(self, ticker: str, data: dict) -> AgentReport:
        """Run financial analysis for a stock."""
        logger.info(f"[FinancialAgent] Analyzing {ticker}...")

        system_prompt = self.build_system_prompt()
        user_message = self.build_user_message(ticker, data)

        try:
            response = await self.llm_client.call_json(
                system_prompt=system_prompt,
                user_message=user_message,
            )
            report = self._parse_llm_response(response, ticker)
            logger.info(f"[FinancialAgent] {ticker} -> score={report.score}, signal={report.signal.value}")
            return report

        except Exception as e:
            logger.error(f"[FinancialAgent] Failed for {ticker}: {e}")
            return AgentReport(
                agent_type=self.agent_type,
                ticker=ticker,
                score=self.config._raw.get("scoring", {}).get("missing_data_default_score", 3.0),
                confidence=0.0,
                reasoning=f"Analysis failed: {str(e)}",
                data_quality="poor",
            )