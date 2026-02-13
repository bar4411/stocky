
"""
Prefilter for EdgeFolio.
Narrows the stock universe from ~500 to ~50-100 candidates using fast,
rule-based filters. No LLM calls — just pandas.
"""

import logging

import yfinance as yf
import pandas as pd

from src.config.config import Config
from src.data.schemas import StockData, PrefilterResult

logger = logging.getLogger(__name__)


class Prefilter:
    """Rule-based prefilter to reduce stock universe before deep analysis."""

    def __init__(self, config: Config):
        self.config = config
        self.pf_config = config.prefilter

    def filter(self, tickers: list[str]) -> list[PrefilterResult]:
        """
        Filter and rank tickers using fast heuristics.
        Returns top N candidates ordered by prefilter score.
        """
        logger.info(f"Pre-filtering {len(tickers)} tickers...")

        # Batch download basic data for speed
        df = self._batch_download(tickers)
        if df.empty:
            logger.warning("No data returned from batch download")
            return []

        # Apply filters
        df = self._apply_filters(df)
        logger.info(f"After filters: {len(df)} tickers remain")

        # Score and rank
        df = self._score(df)

        # Take top N
        top_n = self.config.prefilter_top_n
        df = df.nlargest(top_n, "prefilter_score")
        logger.info(f"Prefilter complete: top {len(df)} candidates selected")

        # Convert to PrefilterResult objects
        results = []
        for _, row in df.iterrows():
            stock = StockData(
                ticker=row["ticker"],
                company_name=row.get("company_name", ""),
                sector=row.get("sector", ""),
                market_cap=row.get("market_cap", 0),
                current_price=row.get("current_price", 0),
                price_52w_high=row.get("52w_high", 0),
                price_52w_low=row.get("52w_low", 0),
                avg_volume=row.get("avg_volume", 0),
            )
            results.append(
                PrefilterResult(
                    stock=stock,
                    prefilter_score=row["prefilter_score"],
                    signals=row.get("signals", []),
                )
            )

        return results

    def _batch_download(self, tickers: list[str]) -> pd.DataFrame:
        """Download basic data for all tickers at once using yfinance."""
        try:
            # Download last 60 days of daily data for all tickers
            data = yf.download(tickers, period="60d", group_by="ticker", progress=False)

            rows = []
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info

                    # Get recent price data
                    hist = stock.history(period="5d")
                    if hist.empty:
                        continue

                    current_price = hist["Close"].iloc[-1]
                    avg_vol_recent = hist["Volume"].mean()

                    # Calculate volume spike vs 60-day average
                    hist_60d = stock.history(period="60d")
                    avg_vol_60d = hist_60d["Volume"].mean() if not hist_60d.empty else avg_vol_recent

                    rows.append({
                        "ticker": ticker,
                        "company_name": info.get("longName", ticker),
                        "sector": info.get("sector", "Unknown"),
                        "industry": info.get("industry", "Unknown"),
                        "market_cap": info.get("marketCap", 0),
                        "current_price": current_price,
                        "52w_high": info.get("fiftyTwoWeekHigh", 0),
                        "52w_low": info.get("fiftyTwoWeekLow", 0),
                        "avg_volume": info.get("averageVolume", 0),
                        "recent_volume": avg_vol_recent,
                        "avg_volume_60d": avg_vol_60d,
                        "pe_ratio": info.get("trailingPE"),
                        "revenue_growth": info.get("revenueGrowth"),
                        "earnings_growth": info.get("earningsGrowth"),
                        "analyst_rec": info.get("recommendationKey", ""),
                    })
                except Exception as e:
                    logger.debug(f"Skipping {ticker}: {e}")
                    continue

            return pd.DataFrame(rows)

        except Exception as e:
            logger.error(f"Batch download failed: {e}")
            return pd.DataFrame()

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply hard filters to remove stocks that don't meet criteria."""
        initial_count = len(df)

        # Min market cap
        min_cap = self.pf_config.min_market_cap_billions * 1e9
        df = df[df["market_cap"] >= min_cap]

        # Min average volume
        df = df[df["avg_volume"] >= self.pf_config.min_avg_volume]

        # Exclude sectors
        if self.pf_config.exclude_sectors:
            df = df[~df["sector"].isin(self.pf_config.exclude_sectors)]

        # Must have a valid price
        df = df[df["current_price"] > 0]

        logger.info(f"Filters removed {initial_count - len(df)} tickers")
        return df

    def _score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score remaining stocks using quick heuristic signals."""
        signals_config = self.pf_config.signals
        df = df.copy()
        df["prefilter_score"] = 5.0  # Baseline
        df["signals"] = [[] for _ in range(len(df))]

        # Signal 1: Volume spike (recent volume >> 60d average)
        vol_threshold = signals_config.get("volume_spike_threshold", 2.0)
        vol_spike = df["recent_volume"] > (df["avg_volume_60d"] * vol_threshold)
        df.loc[vol_spike, "prefilter_score"] += 1.5
        for idx in df[vol_spike].index:
            df.at[idx, "signals"] = df.at[idx, "signals"] + ["volume_spike"]

        # Signal 2: Near 52-week low (potential value)
        low_pct = signals_config.get("near_52w_low_pct", 0.15)
        near_low = (
            (df["current_price"] - df["52w_low"]) / df["52w_low"].replace(0, float("nan"))
        ) <= low_pct
        df.loc[near_low, "prefilter_score"] += 1.0
        for idx in df[near_low].index:
            df.at[idx, "signals"] = df.at[idx, "signals"] + ["near_52w_low"]

        # Signal 3: Positive revenue growth
        pos_growth = df["revenue_growth"].fillna(0) > 0.05
        df.loc[pos_growth, "prefilter_score"] += 1.0
        for idx in df[pos_growth].index:
            df.at[idx, "signals"] = df.at[idx, "signals"] + ["positive_revenue_growth"]

        # Signal 4: Strong earnings growth
        strong_earnings = df["earnings_growth"].fillna(0) > 0.10
        df.loc[strong_earnings, "prefilter_score"] += 1.0
        for idx in df[strong_earnings].index:
            df.at[idx, "signals"] = df.at[idx, "signals"] + ["strong_earnings_growth"]

        # Signal 5: Analyst recommendation is buy or strong buy
        buy_recs = df["analyst_rec"].isin(["buy", "strong_buy"])
        df.loc[buy_recs, "prefilter_score"] += 0.5
        for idx in df[buy_recs].index:
            df.at[idx, "signals"] = df.at[idx, "signals"] + ["analyst_buy"]

        return df