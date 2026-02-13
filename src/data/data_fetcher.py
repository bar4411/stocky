"""
Data Fetcher for EdgeFolio.
Fetches stock data, news, and financial data from various sources.
V1: Pre-fetches all data and passes to agents.
V2 (future): Agents call fetcher methods as tools.
"""

import logging
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import requests

from src.config.config import Config
from src.data.schemas import StockData

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and caches market data from multiple sources."""

    def __init__(self, config: Config):
        self.config = config
        self.n_tickers_fetched = 0
        self._cache: dict = {}

    # ============================================
    # STOCK UNIVERSE
    # ============================================

    def get_tickers(self) -> pd.DataFrame:
        """Fetch current Index tickers list from Wikipedia."""
        try:
            tables = pd.read_html(self.config.tickers_url_source)
            df = tables[0]
            tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
            logger.info(f"Fetched {len(tickers)} {self.config.universe} tickers")
            self.n_tickers_fetched = len(tickers)
            return self._get_tickers_daily_stats(tickers=tickers)
        except Exception as e:
            logger.error(f"Failed to fetch {self.config.universe} list: {e}")
            raise

    def _get_tickers_daily_stats(self, tickers: list[str]) -> pd.DataFrame:
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

    def fetch_all_data(self, filtered_tickers) -> dict[str, dict]:
        """
        Fetch all data for all candidates upfront.
        Returns a dict of ticker -> {stock_data, news, financials, price_history}
        """
        stocks_data = {}

        for i, candidate in enumerate(filtered_tickers):
            ticker = candidate.stock.ticker
            company_name = candidate.stock.company_name
            logger.info(f"  Fetching data [{i+1}/{len(filtered_tickers)}]: {ticker}")

            try:
                stocks_data[ticker] = {
                    "stock_data": candidate.stock,
                    "company_name": company_name,
                    "sector": candidate.stock.sector,
                    "news": self.get_news(ticker, company_name),
                    "financials": self.get_financials(ticker),
                    "price_history": self.get_price_history(
                        ticker,
                        days=self.config.technical_agent.lookback_days
                        if self.config.technical_agent else 90,
                    ),
                }
            except Exception as e:
                logger.warning(f"  Failed to fetch data for {ticker}: {e}")
                continue


        # Validate cached data and return list of tickers with sufficient data.
        # Logs warnings for stocks with missing data.
        valid_data = {}

        for ticker, data in stocks_data.items():
            issues = []

            if not data.get("news"):
                issues.append("no news")
            if not data.get("financials") or not data["financials"].get("ratios"):
                issues.append("no financials")
            if data.get("price_history") is None or (
                hasattr(data["price_history"], "empty") and data["price_history"].empty
            ):
                issues.append("no price history")

            if issues:
                logger.debug(f"  {ticker}: missing {', '.join(issues)} (will still analyze)")

            # Still analyze even with partial data — agents handle missing data gracefully
            if not issues:
                valid_data[ticker] = data

        logger.info(f"  Valid data for: {len(valid_data)}/{len(filtered_tickers)} stocks")
        return valid_data


    # ============================================
    # STOCK DATA (yfinance)
    # ============================================

    def get_stock_data(self, ticker: str) -> StockData:
        """Fetch basic stock info for a single ticker."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return StockData(
                ticker=ticker,
                company_name=info.get("longName", info.get("shortName", ticker)),
                sector=info.get("sector", "Unknown"),
                industry=info.get("industry", "Unknown"),
                market_cap=info.get("marketCap", 0),
                current_price=info.get("currentPrice", info.get("regularMarketPrice", 0)),
                price_52w_high=info.get("fiftyTwoWeekHigh", 0),
                price_52w_low=info.get("fiftyTwoWeekLow", 0),
                avg_volume=info.get("averageVolume", 0),
                pe_ratio=info.get("trailingPE"),
                dividend_yield=info.get("dividendYield"),
            )
        except Exception as e:
            logger.warning(f"Failed to fetch stock data for {ticker}: {e}")
            return StockData(ticker=ticker)

    def get_stock_data_batch(self, tickers: list[str]) -> dict[str, StockData]:
        """Fetch stock data for multiple tickers. Returns dict of ticker -> StockData."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.get_stock_data(ticker)
        return results

    # ============================================
    # PRICE HISTORY (for technical analysis)
    # ============================================

    def get_price_history(
        self, ticker: str, days: int = 90
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for technical analysis."""
        try:
            stock = yf.Ticker(ticker)
            period = f"{days}d" if days <= 365 else f"{days // 365}y"
            df = stock.history(period=period)

            if df.empty:
                logger.warning(f"No price history for {ticker}")
                return pd.DataFrame()

            logger.debug(f"Fetched {len(df)} days of price data for {ticker}")
            return df

        except Exception as e:
            logger.warning(f"Failed to fetch price history for {ticker}: {e}")
            return pd.DataFrame()

    # ============================================
    # NEWS (NewsAPI)
    # ============================================

    def get_news(self, ticker: str, company_name: str = "", days: int = 30) -> list[dict]:
        """
        Fetch recent news articles for a stock.
        Returns list of dicts with: title, description, source, url, published_at
        """
        if not self.config.newsapi_key:
            logger.warning("NewsAPI key not set — skipping news fetch")
            return []

        # Search by ticker and company name for better results
        query = f"{ticker} OR {company_name}" if company_name else ticker
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_date,
                    "sortBy": "relevancy",
                    "pageSize": self.config.news_agent.max_articles_per_stock if self.config.news_agent else 15,
                    "language": "en",
                    "apiKey": self.config.newsapi_key,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            articles = [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                }
                for a in data.get("articles", [])
                if a.get("title") and "[Removed]" not in a.get("title", "")
            ]

            logger.info(f"Fetched {len(articles)} news articles for {ticker}")
            return articles

        except Exception as e:
            logger.warning(f"Failed to fetch news for {ticker}: {e}")
            return []

    # ============================================
    # FINANCIAL DATA (yfinance + SEC EDGAR future)
    # ============================================

    def get_financials(self, ticker: str) -> dict:
        """
        Fetch financial data for fundamental analysis.
        Returns dict with income statement, balance sheet, cash flow, and key ratios.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Key financial ratios
            ratios = {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "free_cash_flow": info.get("freeCashflow"),
                "revenue": info.get("totalRevenue"),
                "ebitda": info.get("ebitda"),
                "total_debt": info.get("totalDebt"),
                "total_cash": info.get("totalCash"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "52w_change": info.get("52WeekChange"),
                "analyst_target": info.get("targetMeanPrice"),
                "analyst_recommendation": info.get("recommendationKey"),
                "num_analysts": info.get("numberOfAnalystOpinions"),
            }

            # Income statement (last 4 quarters)
            income_stmt = {}
            try:
                quarterly = stock.quarterly_income_stmt
                if not quarterly.empty:
                    income_stmt = {
                        "quarterly_revenue": quarterly.loc["Total Revenue"].to_dict()
                        if "Total Revenue" in quarterly.index else {},
                        "quarterly_net_income": quarterly.loc["Net Income"].to_dict()
                        if "Net Income" in quarterly.index else {},
                    }
            except Exception:
                pass

            return {
                "ratios": ratios,
                "income_statement": income_stmt,
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "company_summary": info.get("longBusinessSummary", "")[:500],
            }

        except Exception as e:
            logger.warning(f"Failed to fetch financials for {ticker}: {e}")
            return {"ratios": {}, "income_statement": {}, "sector": "", "industry": ""}