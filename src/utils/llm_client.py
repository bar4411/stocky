"""
LLM Client for EdgeFolio.
Wraps Anthropic/OpenAI API calls with retry logic, cost tracking, and structured output parsing.
"""

import json
import logging
from anthropic import Anthropic
from src.config.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around LLM API calls with cost tracking and retry logic."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if config.provider == "anthropic":
            self.client = Anthropic()  # Uses ANTHROPIC_API_KEY env var
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

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

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=tokens,
                temperature=temp,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            self.call_count += 1
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens

            logger.debug(
                f"LLM call #{self.call_count} | "
                f"Tokens: {response.usage.input_tokens}in/{response.usage.output_tokens}out"
            )

            return response.content[0].text

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
        """Rough cost estimate based on token usage (Claude Sonnet pricing)."""
        # Approximate pricing — update as needed
        input_cost_per_1k = 0.003
        output_cost_per_1k = 0.015
        return (
            (self.total_input_tokens / 1000) * input_cost_per_1k
            + (self.total_output_tokens / 1000) * output_cost_per_1k
        )

    def get_usage_stats(self) -> dict:
        """Return usage statistics for this session."""
        return {
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self._estimate_cost(), 4),
        }