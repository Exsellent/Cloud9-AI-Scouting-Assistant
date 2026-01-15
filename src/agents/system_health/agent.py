import logging
from typing import Dict, Any, List, Optional

import httpx

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("system_health_agent")


def log_method(func):
    """Decorator for logging method calls"""

    async def wrapper(self, *args, **kwargs):
        logger.info(f"{func.__name__} called with args: {args}, kwargs: {kwargs}")
        try:
            result = await func(self, *args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {str(e)}")
            raise

    return wrapper


class SystemHealthAgent(MCPAgent):
    def __init__(self):
        super().__init__("SystemHealth")
        self.llm = LLMClient()

        # List of all agents in the system — we use service names from docker-compose.yml
        # This is critical for working inside the Docker network (localhost does not work between containers)
        self.known_agents = {
            "ScoutingReport": "http://scouting_report:8404/health",
            "DraftCoach": "http://draft_coach:8401/health",
            "CounterPlay": "http://counter_play:8403/health",
            "MatchHistory": "http://match_history:8402/health",
            "StatsTracker": "http://stats_tracker:8407/health",
        }

        self.register_tool("full_system_check", self.full_system_check)
        self.register_tool("quick_ping", self.quick_ping)
        self.register_tool("analyze_anomalies", self.analyze_anomalies)

        logger.info("SystemHealthAgent initialized with monitoring of all agents")

    async def _ping_agent(self, name: str, url: str) -> Dict[str, Any]:
        """Пинг одного агента"""
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(url)
                return {
                    "status": "healthy" if resp.status_code == 200 else "degraded",
                    "status_code": resp.status_code,
                    "response_time_ms": round(resp.elapsed.total_seconds() * 1000, 2),
                    "details": resp.json() if resp.headers.get("content-type", "").startswith(
                        "application/json") else resp.text[:200]
                }
        except httpx.TimeoutException:
            return {"status": "unreachable", "error": "timeout"}
        except httpx.RequestError as e:
            return {"status": "unreachable", "error": str(e)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @log_method
    @metric_counter("system_health")
    async def quick_ping(self) -> Dict[str, Any]:
        """Quick check — are all agents responding?"""
        reasoning: List[ReasoningStep] = []
        reasoning.append(ReasoningStep(
            step_number=1,
            description="Quick system ping requested"
        ))

        results = {}
        for name, url in self.known_agents.items():
            results[name] = await self._ping_agent(name, url)

        healthy_count = sum(1 for r in results.values() if r["status"] == "healthy")
        total = len(results)

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Ping completed",
            output_data={"healthy": healthy_count, "total": total}
        ))

        return {
            "system_status": "healthy" if healthy_count == total else "degraded",
            "healthy_agents": healthy_count,
            "total_agents": total,
            "agent_status": results,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("system_health")
    async def full_system_check(self) -> Dict[str, Any]:
        """Full verification + LLM analysis"""
        reasoning: List[ReasoningStep] = []
        reasoning.append(ReasoningStep(step_number=1, description="Full system check initiated"))

        ping_result = await self.quick_ping()
        reasoning.extend(ping_result.get("reasoning", []))

        agent_status = ping_result["agent_status"]

        if ping_result["system_status"] == "healthy":
            analysis = "All agents are healthy and responsive. System is operating normally."
        else:
            prompt = f"""
You are a system reliability engineer monitoring an AI esports coaching platform.

Current agent health status:
{agent_status}

Identify:
• Which agents are down or degraded
• Possible causes (timeout, error, slow response)
• Recommended actions (restart container, check logs, scale)

Be concise, actionable and professional.
"""
            try:
                analysis = await self.llm.chat(prompt)
            except Exception as e:
                analysis = f"Could not perform AI analysis: {str(e)}. Manual review recommended."

        reasoning.append(ReasoningStep(
            step_number=3,
            description="System health analysis completed",
            output_data={"system_status": ping_result["system_status"]}
        ))

        return {
            "system_status": ping_result["system_status"],
            "summary": analysis.strip(),
            "agent_details": agent_status,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("system_health")
    async def analyze_anomalies(
            self,
            custom_status: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Separate tool: AI analysis of the transmitted status only"""
        data = custom_status or (await self.quick_ping())["agent_status"]

        prompt = f"""
AI Esports Coaching System — Anomaly Detection Report

Current agent status:
{data}

As a senior DevOps engineer for this platform, analyze:
• Critical issues
• Performance anomalies
• Potential cascading failures
• Recovery recommendations

Tone: Technical, urgent if needed, confident.
"""
        try:
            analysis = await self.llm.chat(prompt)
        except Exception as e:
            analysis = "AI analysis unavailable."

        return {
            "analysis": analysis.strip(),
            "input_data": data
        }
