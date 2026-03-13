"""
LLM Client for EdgeFolio.
Wraps Anthropic/Groq/Gemini API calls with retry logic, cost tracking, and structured output parsing.
Supports a fallback chain: if the active provider hits a rate limit, the next provider is tried.
"""

import asyncio
import json
import logging

from anthropic import Anthropic, APIConnectionError as AnthropicConnectionError, RateLimitError as AnthropicRateLimitError
from groq import Groq, APIConnectionError as GroqConnectionError, RateLimitError as GroqRateLimitError
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted as GeminiRateLimitError

from src.config.config import LLMConfig

logger = logging.getLogger(__name__)


# Cost per 1,000 tokens by model name. Add new models here without touching other code.
PRICING_TABLE: dict[str, dict[str, float]] = {
    # Anthropic models
    "claude-sonnet-4-20250514":   {"input_per_1k": 0.003,    "output_per_1k": 0.015},
    "claude-3-5-sonnet-20241022": {"input_per_1k": 0.003,    "output_per_1k": 0.015},
    "claude-3-haiku-20240307":    {"input_per_1k": 0.00025,  "output_per_1k": 0.00125},
    # Groq models
    "llama-3.3-70b-versatile":    {"input_per_1k": 0.00059,  "output_per_1k": 0.00079},
    "llama-3.1-8b-instant":       {"input_per_1k": 0.00005,  "output_per_1k": 0.00008},
    "mixtral-8x7b-32768":         {"input_per_1k": 0.00027,  "output_per_1k": 0.00027},
    # Gemini models
    "gemini-1.5-flash":           {"input_per_1k": 0.000075, "output_per_1k": 0.0003},
    "gemini-1.5-pro":             {"input_per_1k": 0.00125,  "output_per_1k": 0.005},
    "gemini-2.0-flash":           {"input_per_1k": 0.0001,   "output_per_1k": 0.0004},
}

_FALLBACK_PRICING = {"input_per_1k": 0.003, "output_per_1k": 0.015}

# Exceptions that signal a provider-level rate limit (not a transient network error)
_RATE_LIMIT_ERRORS = (GroqRateLimitError, AnthropicRateLimitError, GeminiRateLimitError)
# Exceptions that signal a transient connection error worth retrying
_CONNECTION_ERRORS = (GroqConnectionError, AnthropicConnectionError)


def _build_client(config: LLMConfig):
    """Instantiate the correct SDK client for a given LLMConfig."""
    if config.provider == "anthropic":
        return Anthropic(api_key=config.api_key or None)
    elif config.provider == "groq":
        return Groq(api_key=config.api_key or None)
    elif config.provider == "gemini":
        genai.configure(api_key=config.api_key or None)
        # Return the model name; actual model object is created per-call because
        # google-generativeai GenerativeModel requires the system prompt at construction.
        return config.model
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")


class LLMClient:
    """Wrapper around LLM API calls with cost tracking, retry logic, and provider fallback."""

    def __init__(self, configs: list[LLMConfig]):
        if not configs:
            raise ValueError("LLMClient requires at least one LLMConfig")
        self._configs = configs
        self._clients = [_build_client(c) for c in configs]
        self._active_idx = 0

        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # ------------------------------------------------------------------
    # Active provider helpers
    # ------------------------------------------------------------------

    @property
    def config(self) -> LLMConfig:
        """Active provider config."""
        return self._configs[self._active_idx]

    @property
    def _client(self):
        """Active SDK client (or model name string for Gemini)."""
        return self._clients[self._active_idx]

    def _switch_to_next_provider(self, reason: str) -> bool:
        """Try to advance to the next provider. Returns True if successful, False if exhausted."""
        if self._active_idx + 1 >= len(self._configs):
            return False
        self._active_idx += 1
        next_cfg = self._configs[self._active_idx]
        logger.warning(
            f"{reason} — switching to provider {self._active_idx + 1}/{len(self._configs)}: "
            f"{next_cfg.provider} / {next_cfg.model}"
        )
        return True

    # ------------------------------------------------------------------
    # Pickling support (Anthropic client holds unpicklable objects)
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_clients"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._clients = [_build_client(c) for c in self._configs]

    # ------------------------------------------------------------------
    # API call implementation
    # ------------------------------------------------------------------

    def _do_api_call(
        self,
        temp: float,
        tokens: int,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, int, int]:
        """Execute one synchronous API call against the active provider.
        Returns (text, input_tokens, output_tokens).
        """
        cfg = self.config
        client = self._client

        if cfg.provider == "anthropic":
            response = client.messages.create(
                model=cfg.model,
                max_tokens=tokens,
                temperature=temp,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return (
                response.content[0].text,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

        elif cfg.provider == "groq":
            response = client.chat.completions.create(
                model=cfg.model,
                max_tokens=tokens,
                temperature=temp,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            )
            return (
                response.choices[0].message.content,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        elif cfg.provider == "gemini":
            genai.configure(api_key=cfg.api_key or None)
            model = genai.GenerativeModel(
                model_name=cfg.model,
                system_instruction=system_prompt,
            )
            response = model.generate_content(
                user_message,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=tokens,
                    temperature=temp,
                ),
            )
            text = response.text
            # Gemini usage metadata (may be None on some response types)
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            return text, input_tokens, output_tokens

        raise ValueError(f"Unsupported LLM provider: {cfg.provider}")

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def call(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Make an LLM API call and return the text response.
        On rate-limit errors, automatically switches to the next provider in the chain.

        Args:
            system_prompt: The system/role prompt defining agent behavior
            user_message: The user message with data to analyze
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Raw text response from the LLM
        """
        self._check_limits()

        _retry_delays = [2, 4, 8]
        _max_attempts = len(_retry_delays) + 1

        last_error: Exception | None = None

        # Outer loop: iterate over providers in the fallback chain
        while True:
            temp = temperature or self.config.temperature
            tokens = max_tokens or self.config.max_tokens

            # Inner loop: retry transient connection errors for the current provider
            for attempt in range(_max_attempts):
                try:
                    text, input_tokens, output_tokens = self._do_api_call(
                        temp, tokens, system_prompt, user_message
                    )

                    self.call_count += 1
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens

                    logger.debug(
                        f"LLM call #{self.call_count} [{self.config.provider}/{self.config.model}] | "
                        f"Tokens: {input_tokens}in/{output_tokens}out"
                    )
                    return text

                except _RATE_LIMIT_ERRORS as e:
                    # Rate limit — don't retry this provider, try the next one
                    last_error = e
                    if not self._switch_to_next_provider(
                        f"{self.config.provider} rate limit hit"
                    ):
                        raise RuntimeError(
                            "All LLM providers in the fallback chain have hit their rate limits. "
                            f"Last error: {e}"
                        ) from e
                    break  # Break inner loop, outer loop will retry with new provider

                except _CONNECTION_ERRORS as e:
                    if attempt < _max_attempts - 1:
                        delay = _retry_delays[attempt]
                        logger.warning(
                            f"{self.config.provider.capitalize()} connection error "
                            f"(attempt {attempt + 1}/{_max_attempts}), "
                            f"retrying in {delay}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"LLM call failed after {_max_attempts} attempts: {e}")
                        raise

                except Exception as e:
                    logger.error(f"LLM call failed: {e}")
                    raise
            else:
                # Inner loop completed without break — all retries exhausted for this provider
                # (should not normally happen since connection errors re-raise on last attempt)
                break

    async def call_json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> dict:
        """
        Make an LLM call and parse the response as JSON.
        Adds instruction to respond in JSON format.
        """
        json_instruction = (
            "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "No markdown, no backticks, no explanation outside the JSON."
        )

        response_text = await self.call(
            system_prompt=system_prompt + json_instruction,
            user_message=user_message,
            temperature=temperature,
        )

        # Clean potential markdown formatting
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.error(f"Raw response: {response_text[:500]}")
            raise ValueError(f"LLM did not return valid JSON: {e}")

    def _check_limits(self) -> None:
        """Check if we've exceeded call/cost limits."""
        if self.call_count >= self.config.max_daily_api_calls:
            raise RuntimeError(
                f"Daily API call limit reached: {self.call_count}/{self.config.max_daily_api_calls}"
            )

        estimated_cost = self._estimate_cost()
        if estimated_cost >= self.config.max_cost_per_run_usd:
            raise RuntimeError(
                f"Cost limit reached: ${estimated_cost:.2f}/${self.config.max_cost_per_run_usd}"
            )

    def _estimate_cost(self) -> float:
        """Cost estimate based on token usage, looked up from PRICING_TABLE."""
        pricing = PRICING_TABLE.get(self.config.model, _FALLBACK_PRICING)
        return (
            (self.total_input_tokens / 1000) * pricing["input_per_1k"]
            + (self.total_output_tokens / 1000) * pricing["output_per_1k"]
        )

    def get_usage_stats(self) -> dict:
        """Return usage statistics for this session."""
        return {
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self._estimate_cost(), 4),
            "active_provider": self.config.provider,
            "active_model": self.config.model,
        }
