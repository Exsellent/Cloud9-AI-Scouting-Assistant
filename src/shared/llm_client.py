import asyncio
import logging
import os

import httpx

logger = logging.getLogger("llm_client")


class LLMProviderError(Exception):
    """Base exception for LLM provider failures"""
    pass


class LLMClient:
    """
    Hackathon-optimized LLM Client:
    - Configurable model from .env
    - Retry logic with exponential backoff
    - Proper error handling (no masked errors)
    - Clean logging
    """

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "stub").lower()

        if self.provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise RuntimeError("GROQ_API_KEY is not set")

            self.base_url = "https://api.groq.com/openai/v1/chat/completions"

            # Configurable from .env
            self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
            self.timeout = float(os.getenv("LLM_TIMEOUT", "30"))
            self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))

            logger.info(
                "LLM Client initialized",
                extra={
                    "provider": self.provider,
                    "model": self.model,
                    "timeout": self.timeout,
                    "max_retries": self.max_retries
                }
            )

    async def chat(self, prompt: str) -> str:
        """
        Send chat request to LLM

        Args:
            prompt: User prompt

        Returns:
            LLM response text

        Raises:
            LLMProviderError: On provider errors
        """
        logger.info(
            "LLM request started",
            extra={
                "provider": self.provider,
                "model": self.model if self.provider == "groq" else "stub",
                "prompt_length": len(prompt)
            }
        )

        # Stub mode for development
        if self.provider == "stub":
            return f"[stub] {prompt}"

        if self.provider != "groq":
            raise LLMProviderError(f"Unsupported provider: {self.provider}")

        # Prepare request
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Retry with exponential backoff
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        self.base_url,
                        json=payload,
                        headers=headers
                    )

                    # Handle rate limiting with retry
                    if resp.status_code == 429:
                        if attempt < self.max_retries:
                            wait_time = 2 ** attempt  # Exponential backoff
                            logger.warning(
                                f"Rate limited, retrying in {wait_time}s",
                                extra={"attempt": attempt + 1}
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise LLMProviderError(
                                f"Rate limit exceeded after {self.max_retries} retries"
                            )

                    # Handle auth errors (don't retry)
                    if resp.status_code == 401:
                        raise LLMProviderError("Invalid GROQ_API_KEY")

                    # Raise for other HTTP errors
                    resp.raise_for_status()

                    # Parse response
                    data = resp.json()

                    # Validate response structure
                    choices = data.get("choices", [])
                    if not choices:
                        raise LLMProviderError("Empty response from Groq")

                    content = choices[0].get("message", {}).get("content", "")
                    if not content:
                        raise LLMProviderError("Empty content in Groq response")

                    # Log success
                    logger.info(
                        "LLM response received",
                        extra={
                            "response_length": len(content),
                            "usage_tokens": data.get("usage", {}).get("total_tokens", "unknown"),
                            "attempts": attempt + 1
                        }
                    )

                    return content

            except LLMProviderError:
                raise

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Request failed: {type(e).__name__}, retrying in {wait_time}s",
                        extra={"attempt": attempt + 1, "error": str(e)}
                    )
                    await asyncio.sleep(wait_time)
                    continue

        # Explicit failure
        logger.error(
            "LLM request failed after all retries",
            extra={
                "error_type": type(last_exception).__name__,
                "error": str(last_exception),
                "attempts": self.max_retries + 1
            }
        )
        raise LLMProviderError(
            f"LLM request failed after {self.max_retries + 1} attempts: {last_exception}"
        )

    async def chat_structured(self, prompt: str) -> dict:
        """
        Simple structured output wrapper

        Returns:
            Dict with 'raw' key containing response text
        """
        text = await self.chat(prompt)
        return {"raw": text}

    def get_config(self) -> dict:
        """Get current configuration for debugging"""
        return {
            "provider": self.provider,
            "model": self.model if self.provider == "groq" else "stub",
            "timeout": getattr(self, "timeout", None),
            "max_retries": getattr(self, "max_retries", None)
        }
