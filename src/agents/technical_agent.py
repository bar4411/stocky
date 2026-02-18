"""
Technical Analysis Agent for EdgeFolio.
Analyzes price/volume patterns and technical indicators.
"""

import logging

import pandas as pd

from src.agents.base_agent import BaseAgent
from src.config.config import Config
from src.data.schemas import AgentReport, AgentType
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class TechnicalAgent(BaseAgent):
    """Analyzes technical indicators and price patterns for a given stock."""

    agent_type = AgentType.TECHNICAL

    def __init__(self, config: Config, llm_client: LLMClient):
        super().__init__(config, llm_client)

    def build_system_prompt(self) -> str:
        return """You are a senior technical analyst specializing in stock chart analysis.

Your job is to analyze price data and technical indicators to determine:
1. Trend: Is the stock in an uptrend, downtrend, or consolidation?
2. Momentum: Is momentum building or fading? (RSI, MACD)
3. Support/Resistance: Where are key price levels?
4. Volume: Does volume confirm the price action?
5. Moving Averages: How does price relate to SMA 50 and SMA 200? (Golden/death cross?)

SCORING GUIDELINES:
- 8-10: Strong technical setup — uptrend, bullish crossovers, volume confirmation, breakout imminent
- 6-7.9: Mildly bullish — uptrend intact, some positive indicators
- 4-5.9: Neutral/consolidation — no clear direction, mixed signals
- 2-3.9: Mildly bearish — downtrend, bearish indicators, weak volume
- 0-1.9: Strongly bearish — death cross, breakdown below support, very weak technicals

Focus on actionable signals. A stock near its 200-day SMA with rising volume is more
interesting than one drifting sideways with no volume.

""" + self._build_output_schema()

    def compute_indicators(self, price_df: pd.DataFrame) -> dict:
        """Compute technical indicators from price history."""
        if price_df.empty or len(price_df) < 20:
            return {"error": "Insufficient price data"}

        close = price_df["Close"]
        high = price_df["High"]
        low = price_df["Low"]
        volume = price_df["Volume"]

        indicators = {}

        # Current price
        indicators["current_price"] = round(float(close.iloc[-1]), 2)
        indicators["price_5d_ago"] = round(float(close.iloc[-5]), 2) if len(close) >= 5 else None
        indicators["price_20d_ago"] = round(float(close.iloc[-20]), 2) if len(close) >= 20 else None

        # Simple Moving Averages
        if len(close) >= 50:
            indicators["sma_50"] = round(float(close.rolling(50).mean().iloc[-1]), 2)
        if len(close) >= 200:
            indicators["sma_200"] = round(float(close.rolling(200).mean().iloc[-1]), 2)
        indicators["sma_20"] = round(float(close.rolling(20).mean().iloc[-1]), 2)

        # RSI (14-period)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        indicators["rsi_14"] = round(float(rsi.iloc[-1]), 2) if not rsi.empty else None

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        indicators["macd"] = round(float(macd_line.iloc[-1]), 4)
        indicators["macd_signal"] = round(float(signal_line.iloc[-1]), 4)
        indicators["macd_histogram"] = round(float((macd_line - signal_line).iloc[-1]), 4)

        # Bollinger Bands
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        indicators["bollinger_upper"] = round(float((sma_20 + 2 * std_20).iloc[-1]), 2)
        indicators["bollinger_lower"] = round(float((sma_20 - 2 * std_20).iloc[-1]), 2)

        # Volume analysis
        avg_vol_20 = volume.rolling(20).mean().iloc[-1]
        indicators["avg_volume_20d"] = int(avg_vol_20)
        indicators["latest_volume"] = int(volume.iloc[-1])
        indicators["volume_ratio"] = round(float(volume.iloc[-1] / avg_vol_20), 2) if avg_vol_20 > 0 else 0

        # Price change percentages
        indicators["change_5d_pct"] = round(
            float((close.iloc[-1] / close.iloc[-5] - 1) * 100), 2
        ) if len(close) >= 5 else None
        indicators["change_20d_pct"] = round(
            float((close.iloc[-1] / close.iloc[-20] - 1) * 100), 2
        ) if len(close) >= 20 else None

        # 52-week high/low proximity
        if len(close) >= 252:
            high_52w = high.rolling(252).max().iloc[-1]
            low_52w = low.rolling(252).min().iloc[-1]
            indicators["pct_from_52w_high"] = round(float((close.iloc[-1] / high_52w - 1) * 100), 2)
            indicators["pct_from_52w_low"] = round(float((close.iloc[-1] / low_52w - 1) * 100), 2)

        return indicators

    def build_user_message(self, ticker: str, data: dict) -> str:
        indicators = data.get("indicators", {})
        company_name = data.get("company_name", ticker)

        if "error" in indicators:
            return f"""
Analyze technical indicators for {ticker} ({company_name}).

ERROR: {indicators['error']}
Provide a neutral assessment with low confidence and note "poor" data quality.
"""

        # Format indicators
        indicators_text = ""
        for key, value in indicators.items():
            if value is not None:
                indicators_text += f"  {key}: {value}\n"

        return f"""
Analyze the technical setup for {ticker} ({company_name}).

TECHNICAL INDICATORS (latest values):
{indicators_text}

Key things to evaluate:
- Is price above or below SMA 50 / SMA 200?
- RSI: Overbought (>70), oversold (<30), or neutral?
- MACD: Bullish or bearish crossover?
- Volume: Is volume confirming the trend?
- Bollinger Bands: Is price near upper/lower band?
- Recent momentum: 5-day and 20-day price change direction

Provide your technical analysis and score.
"""

    async def analyze(self, ticker: str, data: dict) -> AgentReport:
        """Run technical analysis for a stock."""
        logger.info(f"[TechnicalAgent] Analyzing {ticker}...")

        # Compute indicators from price data if raw DataFrame provided
        if "price_history" in data and isinstance(data["price_history"], pd.DataFrame):
            data["indicators"] = self.compute_indicators(data["price_history"])

        system_prompt = self.build_system_prompt()
        user_message = self.build_user_message(ticker, data)

        try:
            response = await self.llm_client.call_json(
                system_prompt=system_prompt,
                user_message=user_message,
            )
            report = self._parse_llm_response(response, ticker)
            logger.info(f"[TechnicalAgent] {ticker} -> score={report.score}, signal={report.signal.value}")
            return report

        except Exception as e:
            logger.error(f"[TechnicalAgent] Failed for {ticker}: {e}")
            return AgentReport(
                agent_type=self.agent_type,
                ticker=ticker,
                score=self.config._raw.get("scoring", {}).get("missing_data_default_score", 3.0),
                confidence=0.0,
                reasoning=f"Analysis failed: {str(e)}",
                data_quality="poor",
            )