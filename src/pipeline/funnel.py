"""
Tournament Funnel Pipeline for Stocky.

Replaces the ~320-call sequential analysis (4 LLM calls × 80 stocks) with a
4-round elimination funnel that uses cheap batch LLM calls in the early rounds
and reserves expensive individual deep-dives only for the final survivors.

Approximate LLM call budget (80 stocks in, keep_rate=0.50 per round):
  Round 1 – Technical batch:  ⌈80/15⌉ =  6 calls  → 40 survivors
  Round 2 – News batch:       ⌈40/10⌉ =  4 calls  → 20 survivors
  Round 3 – Financial batch:  ⌈20/5⌉  =  4 calls  → 10 survivors
  Round 4 – Full deep dive:   10 × 4  = 40 calls  → final 10
  ─────────────────────────────────────────────────────────────
  Total                                  ~54 calls  (vs ~320)
"""

import asyncio
import dataclasses
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.agents.agents_manager import AgentsManager
from src.config.config import Config
from src.data.schemas import LeadAgentReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-stock state accumulated across all rounds
# ---------------------------------------------------------------------------

@dataclass
class StockFunnelState:
    """Tracks a single stock's journey through the funnel."""
    ticker: str
    company_name: str
    sector: str
    round1_score: float | None = None   # technical score
    round2_score: float | None = None   # news-sentiment score
    round3_score: float | None = None   # financial score
    eliminated_in: int | None = None    # 1 / 2 / 3 / None (survived all)
    elimination_reason: str = ""


# ---------------------------------------------------------------------------
# Main funnel class
# ---------------------------------------------------------------------------

class TournamentFunnel:
    """
    4-round tournament that progressively eliminates weaker candidates.

    The funnel reuses the existing AgentsManager (and its LLMClient) so all
    LLM calls — batch and individual — are counted through a single client.
    After the funnel completes, expose ``self.agents_manager`` so the pipeline
    can hand it to ``process_output()`` for usage-stat reporting.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        # AgentsManager creates the shared LLMClient internally.
        # Batch rounds borrow the same client so call_count is unified.
        self.agents_manager = AgentsManager(config=config)
        self.llm_client = self.agents_manager.llm_client
        self.technical_agent = self.agents_manager.technical_agent

        # Counters for end-of-run summary
        self._calls_by_round: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        self._start_time: float = 0.0

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    async def run(self, stocks_data: dict[str, dict]) -> list[LeadAgentReport]:
        """
        Run all 4 rounds and return the final LeadAgentReport list.

        Args:
            stocks_data: dict[ticker -> cached data dict] from fetch_tickers()
        """
        import time
        self._start_time = time.time()

        logger.info(
            f"\n{'='*60}\n"
            f"  TOURNAMENT FUNNEL — {len(stocks_data)} stocks entering\n"
            f"{'='*60}"
        )

        # ── Round 1: technical batch ────────────────────────────────────────
        r1_states = await self._run_round1(stocks_data)
        r1_survivors, r1_eliminated = self._eliminate(
            r1_states, "round1_score", self.config.funnel.round1_keep_rate, round_num=1
        )
        self._log_round(1, "Technical", r1_survivors, r1_eliminated)
        self._save_round_results(1, r1_survivors + r1_eliminated)

        # ── Round 2: news sentiment batch ───────────────────────────────────
        r2_states = await self._run_round2(r1_survivors, stocks_data)
        r2_survivors, r2_eliminated = self._eliminate(
            r2_states, "round2_score", self.config.funnel.round2_keep_rate, round_num=2
        )
        self._log_round(2, "News Sentiment", r2_survivors, r2_eliminated)
        self._save_round_results(2, r2_survivors + r2_eliminated)

        # ── Round 3: financial batch ─────────────────────────────────────────
        r3_states = await self._run_round3(r2_survivors, stocks_data)
        r3_survivors, r3_eliminated = self._eliminate(
            r3_states, "round3_score", self.config.funnel.round3_keep_rate, round_num=3
        )
        self._log_round(3, "Financial", r3_survivors, r3_eliminated)
        self._save_round_results(3, r3_survivors + r3_eliminated)

        # ── Round 4: full deep dive ──────────────────────────────────────────
        calls_before_r4 = self.llm_client.call_count
        reports = await self._run_round4(r3_survivors, stocks_data)
        self._calls_by_round[4] = self.llm_client.call_count - calls_before_r4

        # ── Final summary ────────────────────────────────────────────────────
        import time as _time
        duration = _time.time() - self._start_time
        self._log_summary(r3_survivors, reports, duration)
        self._save_summary(r3_survivors, reports, duration)

        return reports

    # -----------------------------------------------------------------------
    # Round implementations
    # -----------------------------------------------------------------------

    async def _run_round1(self, stocks_data: dict[str, dict]) -> list[StockFunnelState]:
        """
        Round 1 — Technical Analysis Batch.

        Computes technical indicators locally (no LLM), then asks the LLM to
        score batches of ``round1_batch_size`` stocks at once.
        """
        logger.info(f"\n[Funnel] ── Round 1: Technical Analysis ──")
        all_tickers = list(stocks_data.keys())
        batch_size = self.config.funnel.round1_batch_size
        states: list[StockFunnelState] = []

        # Pre-compute indicators for all stocks (zero LLM calls)
        indicators_map: dict[str, dict] = {}
        for ticker in all_tickers:
            price_history = stocks_data[ticker].get("price_history")
            indicators_map[ticker] = (
                self.technical_agent.compute_indicators(price_history)
                if price_history is not None
                else {"error": "No price history available"}
            )

        # Score in batches
        calls_before = self.llm_client.call_count
        for batch_start in range(0, len(all_tickers), batch_size):
            batch = all_tickers[batch_start: batch_start + batch_size]
            scores = await self._score_batch_technical(batch, indicators_map, round_num=1)
            for ticker in batch:
                states.append(StockFunnelState(
                    ticker=ticker,
                    company_name=stocks_data[ticker].get("company_name", ticker),
                    sector=stocks_data[ticker].get("sector", ""),
                    round1_score=scores.get(ticker, 5.0),
                ))

        self._calls_by_round[1] = self.llm_client.call_count - calls_before
        logger.info(
            f"[Funnel] Round 1 complete — {len(states)} stocks scored "
            f"in {self._calls_by_round[1]} LLM calls"
        )
        return states

    async def _run_round2(
        self,
        survivors: list[StockFunnelState],
        stocks_data: dict[str, dict],
    ) -> list[StockFunnelState]:
        """
        Round 2 — News Sentiment Batch.

        Feeds compact news headlines (title + source + date, up to 5 articles)
        to the LLM in batches of ``round2_batch_size``.
        """
        logger.info(f"\n[Funnel] ── Round 2: News Sentiment ──")
        batch_size = self.config.funnel.round2_batch_size
        tickers = [s.ticker for s in survivors]

        calls_before = self.llm_client.call_count
        scores_map: dict[str, float] = {}
        for batch_start in range(0, len(tickers), batch_size):
            batch_tickers = tickers[batch_start: batch_start + batch_size]
            batch_data = {
                t: stocks_data[t].get("news", []) for t in batch_tickers
            }
            batch_names = {
                t: stocks_data[t].get("company_name", t) for t in batch_tickers
            }
            scores = await self._score_batch_news(
                batch_tickers, batch_data, batch_names, round_num=2
            )
            scores_map.update(scores)

        self._calls_by_round[2] = self.llm_client.call_count - calls_before
        logger.info(
            f"[Funnel] Round 2 complete — {len(tickers)} stocks scored "
            f"in {self._calls_by_round[2]} LLM calls"
        )

        # Attach scores to state objects (carry forward round1_score)
        for state in survivors:
            state.round2_score = scores_map.get(state.ticker, 5.0)
        return survivors

    async def _run_round3(
        self,
        survivors: list[StockFunnelState],
        stocks_data: dict[str, dict],
    ) -> list[StockFunnelState]:
        """
        Round 3 — Financial Analysis Batch.

        Feeds key financial ratios (PE, ROE, growth, margins, debt) to the LLM
        in batches of ``round3_batch_size``.
        """
        logger.info(f"\n[Funnel] ── Round 3: Financial Analysis ──")
        batch_size = self.config.funnel.round3_batch_size
        tickers = [s.ticker for s in survivors]

        calls_before = self.llm_client.call_count
        scores_map: dict[str, float] = {}
        for batch_start in range(0, len(tickers), batch_size):
            batch_tickers = tickers[batch_start: batch_start + batch_size]
            batch_data = {
                t: stocks_data[t].get("financials", {}) for t in batch_tickers
            }
            scores = await self._score_batch_financial(
                batch_tickers, batch_data, round_num=3
            )
            scores_map.update(scores)

        self._calls_by_round[3] = self.llm_client.call_count - calls_before
        logger.info(
            f"[Funnel] Round 3 complete — {len(tickers)} stocks scored "
            f"in {self._calls_by_round[3]} LLM calls"
        )

        for state in survivors:
            state.round3_score = scores_map.get(state.ticker, 5.0)
        return survivors

    async def _run_round4(
        self,
        survivors: list[StockFunnelState],
        stocks_data: dict[str, dict],
    ) -> list[LeadAgentReport]:
        """
        Round 4 — Full Individual Deep Dive.

        Reuses the existing AgentsManager._analyze_single_stock() which makes
        4 LLM calls per stock (news + financial + technical + lead synthesis).
        """
        logger.info(f"\n[Funnel] ── Round 4: Full Deep Dive ({len(survivors)} stocks) ──")
        reports: list[LeadAgentReport] = []

        semaphore = asyncio.Semaphore(self.config.parallel_workers)

        async def _analyze(state: StockFunnelState) -> LeadAgentReport | None:
            async with semaphore:
                try:
                    report = await self.agents_manager._analyze_single_stock(
                        state.ticker, stocks_data[state.ticker]
                    )
                    logger.info(
                        f"[Round 4] {state.ticker} → "
                        f"final_score={report.final_score:.1f}, signal={report.signal.value}"
                    )
                    return report
                except Exception as e:
                    logger.error(f"[Round 4] {state.ticker} deep dive failed: {e}")
                    return None

        tasks = [_analyze(s) for s in survivors]
        results = await asyncio.gather(*tasks)

        for result in results:
            if isinstance(result, LeadAgentReport):
                reports.append(result)

        logger.info(
            f"[Funnel] Round 4 complete — {len(reports)} stocks fully analyzed"
        )
        return reports

    # -----------------------------------------------------------------------
    # Batch LLM scoring helpers
    # -----------------------------------------------------------------------

    async def _score_batch_technical(
        self,
        batch_tickers: list[str],
        indicators_map: dict[str, dict],
        round_num: int,
    ) -> dict[str, float]:
        """Build and fire the Round 1 technical scoring prompt."""
        system_prompt = (
            "You are a quantitative technical analyst. "
            "Score each stock's technical setup using ONLY the provided numerical "
            "indicators. Be concise and objective. Return JSON only."
        )

        # Build compact per-stock lines
        stock_lines = []
        for ticker in batch_tickers:
            ind = indicators_map.get(ticker, {})
            if "error" in ind:
                stock_lines.append(f"{ticker}: insufficient data")
                continue
            parts = []
            if ind.get("rsi_14") is not None:
                parts.append(f"RSI={ind['rsi_14']:.1f}")
            if ind.get("change_5d_pct") is not None:
                parts.append(f"5d={ind['change_5d_pct']:+.1f}%")
            if ind.get("change_20d_pct") is not None:
                parts.append(f"20d={ind['change_20d_pct']:+.1f}%")
            if ind.get("macd_histogram") is not None:
                parts.append(f"macd_hist={ind['macd_histogram']:.2f}")
            if ind.get("volume_ratio") is not None:
                parts.append(f"vol_ratio={ind['volume_ratio']:.2f}")
            if ind.get("pct_from_52w_high") is not None:
                parts.append(f"pct_52wH={ind['pct_from_52w_high']:+.1f}%")
            if ind.get("pct_from_52w_low") is not None:
                parts.append(f"pct_52wL={ind['pct_from_52w_low']:+.1f}%")
            stock_lines.append(f"{ticker}: {', '.join(parts)}" if parts else f"{ticker}: no data")

        user_message = (
            "Score each stock's technical setup on a scale of 1-10. "
            "Higher = stronger bullish technical setup.\n\n"
            "Stocks:\n" + "\n".join(stock_lines) + "\n\n"
            'Return JSON: {"scores": [{"ticker": "AAPL", "score": 6.5, '
            '"reason": "one-sentence reason"}, ...]}'
        )

        return await self._call_batch_llm(system_prompt, user_message, batch_tickers, round_num)

    async def _score_batch_news(
        self,
        batch_tickers: list[str],
        news_map: dict[str, list],
        names_map: dict[str, str],
        round_num: int,
    ) -> dict[str, float]:
        """Build and fire the Round 2 news sentiment scoring prompt."""
        system_prompt = (
            "You are a financial news sentiment analyst. "
            "Score each stock's recent news sentiment (1-10) using ONLY the "
            "headlines and dates provided. Higher = more bullish. Return JSON only."
        )

        stock_blocks = []
        for ticker in batch_tickers:
            name = names_map.get(ticker, ticker)
            articles = news_map.get(ticker, [])
            if not articles:
                stock_blocks.append(f"{ticker} ({name}): no recent news")
                continue
            # Up to 5 most recent articles; compact format
            lines = []
            for art in articles[:5]:
                title = art.get("title", "N/A")[:80]
                source = art.get("source", "N/A")
                date = art.get("published_at", "N/A")[:10]
                lines.append(f'  - "{title}" | {source} | {date}')
            stock_blocks.append(f"{ticker} ({name}):\n" + "\n".join(lines))

        user_message = (
            "Score each stock's news sentiment (1-10). Higher = more bullish.\n\n"
            + "\n".join(stock_blocks) + "\n\n"
            'Return JSON: {"scores": [{"ticker": "AAPL", "score": 7.5, '
            '"reason": "one-sentence reason"}, ...]}'
        )

        return await self._call_batch_llm(system_prompt, user_message, batch_tickers, round_num)

    async def _score_batch_financial(
        self,
        batch_tickers: list[str],
        financials_map: dict[str, dict],
        round_num: int,
    ) -> dict[str, float]:
        """Build and fire the Round 3 financial scoring prompt."""
        system_prompt = (
            "You are a financial analyst. Score each stock's financial health "
            "and valuation (1-10) using only the key ratios provided. "
            "Higher = stronger fundamentals. Return JSON only."
        )

        stock_lines = []
        for ticker in batch_tickers:
            fin = financials_map.get(ticker, {})
            ratios = fin.get("ratios", {}) if fin else {}
            parts = []
            _add = lambda key, label, fmt=".1f": (
                parts.append(f"{label}={ratios[key]:{fmt}}")
                if ratios.get(key) is not None else None
            )
            _add("pe_ratio", "PE")
            _add("forward_pe", "fwd_PE")
            _add("peg_ratio", "PEG")
            _add("roe", "ROE", ".1%") if isinstance(ratios.get("roe"), float) else None
            _add("debt_to_equity", "debt_eq")
            if ratios.get("revenue_growth") is not None:
                parts.append(f"rev_growth={ratios['revenue_growth']:+.1%}")
            if ratios.get("profit_margin") is not None:
                parts.append(f"margin={ratios['profit_margin']:.1%}")
            stock_lines.append(
                f"{ticker}: {', '.join(parts)}" if parts else f"{ticker}: no financial data"
            )

        user_message = (
            "Score each stock's financial health (1-10). Higher = stronger fundamentals.\n\n"
            "Stocks:\n" + "\n".join(stock_lines) + "\n\n"
            'Return JSON: {"scores": [{"ticker": "AAPL", "score": 7.8, '
            '"reason": "one-sentence reason"}, ...]}'
        )

        return await self._call_batch_llm(system_prompt, user_message, batch_tickers, round_num)

    # -----------------------------------------------------------------------
    # Shared batch LLM caller + parser
    # -----------------------------------------------------------------------

    async def _call_batch_llm(
        self,
        system_prompt: str,
        user_message: str,
        batch_tickers: list[str],
        round_num: int,
    ) -> dict[str, float]:
        """
        Fire one batch LLM call and return dict[ticker -> score].

        On *any* error (API failure, malformed JSON, missing tickers), logs the
        error and assigns a neutral score of 5.0 — stocks are never silently
        dropped from the funnel.
        """
        try:
            response = await self.llm_client.call_json(system_prompt, user_message)
            return self._parse_batch_scores(response, batch_tickers, round_num)
        except Exception as e:
            logger.error(
                f"[Round {round_num}] Batch LLM call failed: {e}. "
                f"Assigning neutral 5.0 to {len(batch_tickers)} stocks: {batch_tickers}"
            )
            return {ticker: 5.0 for ticker in batch_tickers}

    def _parse_batch_scores(
        self,
        response: dict,
        batch_tickers: list[str],
        round_num: int,
    ) -> dict[str, float]:
        """
        Parse the ``{"scores": [...]}`` JSON response into a dict[ticker -> float].

        Handles:
        - Wrong type for ``scores`` key
        - Missing ``ticker`` or ``score`` fields per item
        - Tickers the LLM omitted (assigned 5.0 with a warning)
        - Any other parse exception (all assigned 5.0, full batch logged)
        """
        try:
            scores_list = response.get("scores", [])
            if not isinstance(scores_list, list):
                raise ValueError(
                    f"Expected list under 'scores', got {type(scores_list).__name__}"
                )
            result: dict[str, float] = {}
            batch_set = set(batch_tickers)
            for item in scores_list:
                if not isinstance(item, dict):
                    continue
                ticker = item.get("ticker")
                score = item.get("score")
                if ticker in batch_set and score is not None:
                    try:
                        result[ticker] = max(0.0, min(10.0, float(score)))
                    except (TypeError, ValueError):
                        logger.warning(
                            f"[Round {round_num}] Invalid score value for {ticker}: "
                            f"{score!r}, assigning 5.0"
                        )
                        result[ticker] = 5.0

            # Fill any tickers the LLM omitted
            for ticker in batch_tickers:
                if ticker not in result:
                    logger.warning(
                        f"[Round {round_num}] No score returned for {ticker}, assigning 5.0"
                    )
                    result[ticker] = 5.0

            return result

        except Exception as e:
            logger.error(
                f"[Round {round_num}] Score parse failed: {e}. "
                f"Assigning 5.0 to all {len(batch_tickers)} stocks."
            )
            return {ticker: 5.0 for ticker in batch_tickers}

    # -----------------------------------------------------------------------
    # Elimination
    # -----------------------------------------------------------------------

    def _eliminate(
        self,
        states: list[StockFunnelState],
        score_attr: str,
        keep_rate: float,
        round_num: int,
    ) -> tuple[list[StockFunnelState], list[StockFunnelState]]:
        """
        Sort by score descending, keep top ``keep_rate`` fraction, mark the
        rest as eliminated. Returns (survivors, eliminated).

        ``keep_rate`` is applied to the number of stocks that *entered this
        round*, not a fixed absolute count — so the funnel adapts if fewer
        stocks pass the upstream prefilter.
        """
        # 1. Sort all entrants by their score (highest first)
        states.sort(key=lambda s: getattr(s, score_attr) or 0.0, reverse=True)

        # 2. Compute cutoff as a rate of the actual entrant count
        n_keep = max(1, round(len(states) * keep_rate))

        # 3. Explicit partition
        survivors = states[:n_keep]
        eliminated = states[n_keep:]

        # 4. Mark eliminated stocks (never silently dropped)
        for s in eliminated:
            s.eliminated_in = round_num
            s.elimination_reason = (
                f"Round {round_num} {score_attr}={getattr(s, score_attr):.1f} "
                f"below cutoff (kept top {keep_rate:.0%} of {len(states)})"
            )

        logger.info(
            f"[Funnel] Round {round_num} elimination: "
            f"{len(survivors)} survivors / {len(eliminated)} eliminated "
            f"(keep_rate={keep_rate:.0%}, n_in={len(states)}, n_keep={n_keep})"
        )
        return survivors, eliminated

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def _log_round(
        self,
        round_num: int,
        name: str,
        survivors: list[StockFunnelState],
        eliminated: list[StockFunnelState],
    ) -> None:
        score_attr = f"round{round_num}_score"
        logger.info(f"\n[Funnel] === Round {round_num} ({name}) Results ===")
        logger.info(
            f"  Survivors ({len(survivors)}): "
            + ", ".join(
                f"{s.ticker}={getattr(s, score_attr):.1f}" for s in survivors
            )
        )
        if eliminated:
            logger.info(
                f"  Eliminated ({len(eliminated)}): "
                + ", ".join(
                    f"{s.ticker}={getattr(s, score_attr):.1f}" for s in eliminated
                )
            )

    def _log_summary(
        self,
        final_states: list[StockFunnelState],
        reports: list[LeadAgentReport],
        duration: float,
    ) -> None:
        total_calls = self.llm_client.call_count
        logger.info(f"\n{'='*60}")
        logger.info("  FUNNEL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"  Total LLM calls : {total_calls}")
        for rnd, cnt in self._calls_by_round.items():
            logger.info(f"    Round {rnd}        : {cnt} calls")
        logger.info(f"  Total duration  : {duration:.1f}s")
        logger.info(f"  Final stocks    : {len(reports)}")
        for r in sorted(reports, key=lambda x: x.final_score, reverse=True):
            logger.info(
                f"    {r.ticker:6s} | score={r.final_score:.1f} "
                f"| signal={r.signal.value}"
            )
        logger.info(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Disk persistence
    # -----------------------------------------------------------------------

    def _save_round_results(
        self, round_num: int, states: list[StockFunnelState]
    ) -> None:
        """Save all states (survivors + eliminated) for a round to JSON."""
        out_dir = Path(self.config.output_dir) / "funnel"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"round{round_num}_results.json"
        data = [dataclasses.asdict(s) for s in states]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"[Funnel] Round {round_num} results saved → {path}")

    def _save_summary(
        self,
        final_states: list[StockFunnelState],
        reports: list[LeadAgentReport],
        duration: float,
    ) -> None:
        """Save a human-readable summary of the entire funnel run."""
        out_dir = Path(self.config.output_dir) / "funnel"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "summary.json"

        summary = {
            "run_date": datetime.utcnow().isoformat(),
            "total_llm_calls": self.llm_client.call_count,
            "calls_by_round": self._calls_by_round,
            "total_duration_seconds": round(duration, 2),
            "llm_usage": self.llm_client.get_usage_stats(),
            "final_stocks": [
                {
                    "ticker": r.ticker,
                    "company_name": r.company_name,
                    "final_score": r.final_score,
                    "signal": r.signal.value,
                    "confidence": r.confidence,
                }
                for r in sorted(reports, key=lambda x: x.final_score, reverse=True)
            ],
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"[Funnel] Summary saved → {path}")
