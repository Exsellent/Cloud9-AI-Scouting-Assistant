import logging
import os
from typing import Dict, Any

import httpx

logger = logging.getLogger("llm_client")


class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "stub").lower()

        if self.provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise RuntimeError("GROQ_API_KEY is not set")
            self.base_url = "https://api.groq.com/openai/v1/chat/completions"
            self.model = "llama-3.3-70b-versatile"

    async def chat(self, prompt: str) -> str:

        logger.info("LLM request started", extra={
            "provider": self.provider,
            "model": self.model if self.provider == "groq" else "stub",
            "prompt_length": len(prompt)
        })

        if self.provider == "stub":
            return f"[stub] {prompt}"

        if self.provider != "groq":
            return f"[unsupported provider] {self.provider}"

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

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self.base_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]

                logger.info("LLM response received", extra={
                    "response_length": len(content),
                    "usage_tokens": data.get("usage", {}).get("total_tokens", "unknown"),
                    "usage_prompt_tokens": data.get("usage", {}).get("prompt_tokens", "unknown"),
                    "usage_completion_tokens": data.get("usage", {}).get("completion_tokens", "unknown")
                })

                return content
        except Exception as e:
            logger.error("LLM request failed", extra={"error_type": type(e).__name__, "error": str(e)})
            return f"[LLM error] {str(e)}"

    async def chat_structured(self, prompt: str) -> Dict[str, Any]:
        text = await self.chat(prompt)
        return {"raw": text}
