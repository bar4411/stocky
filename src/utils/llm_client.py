"""
LLM Client for EdgeFolio.
Wraps Anthropic/Groq API calls with retry logic, cost tracking, and structured output parsing.
"""

import asyncio
import json
import logging

from anthropic import Anthropic, APIConnectionError as AnthropicConnectionError
from groq import Groq, APIConnectionError as GroqConnectionError

from src.config.config import LLMConfig

logger = logging.getLogger(__name__)


# Cost per 1,000 tokens by model name. Add new models here without touching other code.
PRICING_TABLE: dict[str, dict[str, float]] = {
    # Anthropic models
    "claude-sonnet-4-20250514":   {"input_per_1k": 0.003,   "output_per_1k": 0.015},
    "claude-3-5-sonnet-20241022": {"input_per_1k": 0.003,   "output_per_1k": 0.015},
    "claude-3-haiku-20240307":    {"input_per_1k": 0.00025, "output_per_1k": 0.00125},
    # Groq models
    "llama-3.3-70b-versatile":    {"input_per_1k": 0.00059, "output_per_1k": 0.00079},
    "llama-3.1-8b-instant":       {"input_per_1k": 0.00005, "output_per_1k": 0.00008},
    "mixtral-8x7b-32768":         {"input_per_1k": 0.00027, "output_per_1k": 0.00027},
}

_FALLBACK_PRICING = {"input_per_1k": 0.003, "output_per_1k": 0.015}


class LLMClient:
    """Wrapper around LLM API calls with cost tracking and retry logic."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if config.provider == "anthropic":
            self.client = Anthropic(api_key=config.api_key or None)
            self._connection_error_cls = AnthropicConnectionError
        elif config.provider == "groq":
            self.client = Groq(api_key=config.api_key or None)
            self._connection_error_cls = GroqConnectionError
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

    def _do_api_call(
        self,
        temp: float,
        tokens: int,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, int, int]:
        """Execute one synchronous API call. Returns (text, input_tokens, output_tokens)."""
        if self.config.provider == "anthropic":
            response = self.client.messages.create(
                model=self.config.model,
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

        elif self.config.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.config.model,
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

        raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    async def call(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Make an LLM API call and return the text response.

        Args:
            system_prompt: The system/role prompt defining agent behavior
            user_message: The user message with data to analyze
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Raw text response from the LLM
        """
        self._check_limits()

        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        _retry_delays = [2, 4, 8]
        _max_attempts = len(_retry_delays) + 1

        for attempt in range(_max_attempts):
            try:
                text, input_tokens, output_tokens = self._do_api_call(
                    temp, tokens, system_prompt, user_message
                )

                self.call_count += 1
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens

                logger.debug(
                    f"LLM call #{self.call_count} | "
                    f"Tokens: {input_tokens}in/{output_tokens}out"
                )

                return text

            except self._connection_error_cls as e:
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
        }
