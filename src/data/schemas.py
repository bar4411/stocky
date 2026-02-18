"""
Data schemas for EdgeFolio.
Defines structured data models for agents, reports, and pipeline outputs.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ============================================
# ENUMS
# ============================================

class AgentType(str, Enum):
    NEWS = "news"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    LEAD = "lead"


class Signal(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class RiskLevel(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# ============================================
# STOCK DATA
# ============================================

class StockData(BaseModel):
    """Raw stock data for a single ticker."""
    ticker: str
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: float = 0.0
    current_price: float = 0.0
    price_52w_high: float = 0.0
    price_52w_low: float = 0.0
    avg_volume: float = 0.0
    pe_ratio: float | None = None
    dividend_yield: float | None = None


class PrefilterResult(BaseModel):
    """Result of the prefilter step — a candidate stock with basic signals."""
    stock: StockData
    prefilter_score: float = Field(ge=0, le=10, description="Quick scoring based on simple signals")
    signals: list[str] = Field(default_factory=list, description="Which signals triggered")


# ============================================
# AGENT REPORTS
# ============================================

class AgentReport(BaseModel):
    """Output from a single agent's analysis of one stock."""
    agent_type: AgentType
    ticker: str
    score: float = Field(ge=0, le=10, description="Agent's score for this stock")
    signal: Signal = Signal.NEUTRAL
    confidence: float = Field(ge=0, le=1, description="How confident the agent is (0-1)")
    reasoning: str = Field(description="Detailed explanation of the score")
    key_factors: list[str] = Field(default_factory=list, description="Top 3-5 factors driving the score")
    risks: list[str] = Field(default_factory=list, description="Key risks identified")
    data_quality: str = Field(default="good", description="Quality of data available: good, partial, poor")
    raw_data_summary: dict = Field(default_factory=dict, description="Summary of raw data used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LeadAgentReport(BaseModel):
    """Final synthesized report from the Lead Agent."""
    ticker: str
    company_name: str = ""
    sector: str = ""
    final_score: float = Field(ge=0, le=10)
    signal: Signal = Signal.NEUTRAL
    confidence: float = Field(ge=0, le=1)

    # Breakdown
    news_score: float | None = None
    financial_score: float | None = None
    technical_score: float | None = None

    # Synthesis
    investment_thesis: str = Field(description="Why this stock is recommended (or not)")
    key_catalysts: list[str] = Field(default_factory=list, description="What could drive the stock up")
    key_risks: list[str] = Field(default_factory=list, description="What could go wrong")
    agent_agreement: str = Field(default="", description="Do agents agree or conflict?")

    # Sub-agent reports
    agent_reports: list[AgentReport] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================
# PIPELINE OUTPUT
# ============================================

class DiscoveryOutput(BaseModel):
    """Final output of the Stock Discovery Engine."""
    run_date: datetime = Field(default_factory=datetime.utcnow)
    universe: str = "sp500"
    total_stocks_scanned: int = 0
    candidates_analyzed: int = 0
    risk_profile: RiskLevel = RiskLevel.MODERATE

    # The top K recommendations
    top_picks: list[LeadAgentReport] = Field(default_factory=list)

    # Run metadata
    pipeline_mode: str = "llm"
    total_llm_calls: int = 0
    total_duration_seconds: float = 0.0

    def summary(self) -> str:
        """Print a human-readable summary."""
        lines = [
            f"\n{'='*60}",
            f"  EDGEFOLIO STOCK DISCOVERY — {self.run_date.strftime('%Y-%m-%d')}",
            f"{'='*60}",
            f"  Universe: {self.universe} | Scanned: {self.total_stocks_scanned} | Analyzed: {self.candidates_analyzed}",
            f"  Risk Profile: {self.risk_profile.value} | Mode: {self.pipeline_mode}",
            f"{'='*60}",
            f"  TOP {len(self.top_picks)} RECOMMENDATIONS",
            f"{'='*60}",
        ]
        for i, pick in enumerate(self.top_picks, 1):
            lines.append(
                f"  #{i:2d} | {pick.ticker:6s} | Score: {pick.final_score:.1f}/10 | "
                f"Signal: {pick.signal.value:12s} | {pick.company_name}"
            )
            lines.append(f"       Thesis: {pick.investment_thesis[:80]}...")
            lines.append("")

        lines.append(f"  Duration: {self.total_duration_seconds:.1f}s | LLM Calls: {self.total_llm_calls}")
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)