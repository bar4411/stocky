"""
Configuration loader for EdgeFolio.
Loads config.yaml and resolves environment variables for API keys.
"""

import os
import time

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# Default config path - can be overridden by APP_CONFIG_PATH env var
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "conf" / "conf.yaml"
CONFIG_PATH = os.environ.get('APP_CONFIG_PATH', str(DEFAULT_CONFIG_PATH))


@dataclass
class AgentConfig:
    enabled: bool
    weight: float
    lookback_days: int = 30
    max_articles_per_stock: int = 15
    reports: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)


@dataclass
class LLMConfig:
    provider: str
    model: str
    max_tokens: int
    temperature: float
    max_daily_api_calls: int
    max_cost_per_run_usd: float
    api_key: str = ""


@dataclass
class PrefilterConfig:
    min_market_cap_billions: float
    min_avg_volume: int
    exclude_sectors: list[str]
    signals: dict


class Config:
    """Main configuration object for EdgeFolio."""

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """Load configuration from YAML file."""
        if path:
            config_path = Path(path)
        else:
            config_path = Path(CONFIG_PATH)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        # Handle feature_1 wrapper if present
        if "feature_1" in raw:
            raw = raw["feature_1"]

        config = cls(_raw=raw)
        config._parse(raw)
        return config
    
    def __init__(self, _raw: dict = None):
        """Initialize Config. Use Config.load() to create from file."""
        if _raw is None:
            raise ValueError("Use Config.load() to create Config from file, or pass _raw dict")
        self._raw = _raw or {}
        self.start_time = time.time()
        # Initialize defaults
        self.universe = "sp500"
        self.top_k = 10
        self.prefilter_top_n = 80
        self.pipeline_mode = "llm"
        self.parallel_workers = 5
        self.news_agent = None
        self.financial_agent = None
        self.technical_agent = None
        self.llm = None
        self.prefilter = None
        self.lead_agent_mode = "llm_reasoning"
        self.min_confidence = 5.0
        self.risk_profile = "moderate"
        self.output_format = "json"
        self.save_agent_reports = True
        self.output_dir = "outputs"
        self.timestamp_format = "%Y-%m-%d_%H-%M-%S"
        self.anthropic_api_key = ""
        self.newsapi_key = ""
        self.finnhub_key = ""
        self.tickers_url_source = None

    def _parse(self, raw: dict) -> None:
        """Parse raw YAML dict into typed config."""
        self.start_time = time.time()
        # Market
        market = raw.get("market", {})
        self.universe = market.get("universe", "sp500")
        self.top_k = market.get("top_k", 10)
        self.prefilter_top_n = market.get("prefilter_top_n", 80)
        self.tickers_url_source = market.get("tickers_url_source")

        # Pipeline
        pipeline = raw.get("pipeline", {})
        self.pipeline_mode = pipeline.get("mode", "llm")
        self.parallel_workers = pipeline.get("parallel_workers", 5)

        # Agents
        agents = raw.get("agents", {})
        self.news_agent = self._parse_agent(agents.get("news", {}))
        self.financial_agent = self._parse_agent(agents.get("financial", {}))
        self.technical_agent = self._parse_agent(agents.get("technical", {}))

        # API Keys (resolve env vars) — parsed before LLMConfig so api_key is available
        keys = raw.get("api_keys", {})
        self.anthropic_api_key = self._resolve_env(keys.get("anthropic", ""))
        self.newsapi_key = self._resolve_env(keys.get("newsapi", ""))
        self.finnhub_key = self._resolve_env(keys.get("finnhub", ""))

        # LLM
        llm = raw.get("llm", {})
        self.llm = LLMConfig(
            provider=llm.get("provider", "anthropic"),
            model=llm.get("model", "claude-sonnet-4-20250514"),
            max_tokens=llm.get("max_tokens", 2000),
            temperature=llm.get("temperature", 0.3),
            max_daily_api_calls=llm.get("max_daily_api_calls", 500),
            max_cost_per_run_usd=llm.get("max_cost_per_run_usd", 5.0),
            api_key=self.anthropic_api_key,
        )

        # Prefilter
        pf = raw.get("prefilter", {})
        self.prefilter = PrefilterConfig(
            min_market_cap_billions=pf.get("min_market_cap_billions", 5.0),
            min_avg_volume=pf.get("min_avg_volume", 500000),
            exclude_sectors=pf.get("exclude_sectors", []),
            signals=pf.get("signals", {}),
        )

        # Lead agent
        lead = raw.get("lead_agent", {})
        self.lead_agent_mode = lead.get("mode", "llm_reasoning")
        self.min_confidence = lead.get("min_confidence", 5.0)

        # Risk
        self.risk_profile = raw.get("risk_profile", {}).get("level", "moderate")

        # Output
        output = raw.get("output", {})
        self.output_format = output.get("format", "json")
        self.save_agent_reports = output.get("save_agent_reports", True)
        self.output_dir = output.get("save_dir", "outputs")
        self.timestamp_format = output.get("timestamp_format", "%Y-%m-%d_%H-%M-%S")

    def _parse_agent(self, agent_dict: dict) -> AgentConfig:
        """Parse a single agent config section."""
        return AgentConfig(
            enabled=agent_dict.get("enabled", False),
            weight=agent_dict.get("weight", 0.0),
            lookback_days=agent_dict.get("lookback_days", 30),
            max_articles_per_stock=agent_dict.get("max_articles_per_stock", 15),
            reports=agent_dict.get("reports", []),
            metrics=agent_dict.get("metrics", []),
            indicators=agent_dict.get("indicators", []),
        )

    @staticmethod
    def _resolve_env(value: str) -> str:
        """Resolve ${ENV_VAR} to actual environment variable value."""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, "")
        return value

    def get_agent_weights(self) -> dict[str, float]:
        """Return normalized weights for enabled agents."""
        weights = {}
        if self.news_agent and self.news_agent.enabled:
            weights["news"] = self.news_agent.weight
        if self.financial_agent and self.financial_agent.enabled:
            weights["financial"] = self.financial_agent.weight
        if self.technical_agent and self.technical_agent.enabled:
            weights["technical"] = self.technical_agent.weight

        # Normalize so weights sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights