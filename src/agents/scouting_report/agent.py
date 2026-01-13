import json
import logging
from typing import List

from shared.grid_client import GRIDClient
from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("scouting_report_agent")


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


class ScoutingReportAgent(MCPAgent):
    def __init__(self):
        super().__init__("ScoutingReport")
        self.llm = LLMClient()
        self.grid = GRIDClient()

        self.register_tool("generate_scout_report", self.generate_scout_report)

        logger.info("ScoutingReportAgent initialized")

    def _is_invalid_response(self, response: str) -> bool:
        """Check if LLM response is stub or error"""
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "for more information check", "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    @log_method
    @metric_counter("scouting_report")
    async def generate_scout_report(
            self,
            opponent_team: str,
            game: str = "valorant",  
            recent_matches: int = 5
    ):
        """Generate automated scouting report for opponent team"""
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Scouting report requested",
            input_data={
                "opponent": opponent_team,
                "game": game,
                "matches": recent_matches
            }
        ))

        # === receive match data via the GRID===
        matches_data = await self.grid.get_recent_matches(
            team_name=opponent_team,
            game=game,
            limit=recent_matches
        )

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Fetched match data from GRID (or demo fallback)",
            output_data={"matches_found": len(matches_data)}
        ))

        # === Creating an prompt for LLM ===
        if game == "valorant":
            prompt = f"""
You are a professional Valorant esports analyst preparing a concise scouting report.

Opponent: {opponent_team}
Matches analyzed: {len(matches_data)}

Raw match data:
{json.dumps(matches_data, indent=2)}

Generate a coach-friendly report (under 300 words) covering:
• Preferred agent compositions (most frequent 3–5 agents)
• Map-specific defaults and site setups
• Key players and their roles/tendencies
• Strengths (what they excel at)
• Weaknesses (exploitable patterns)
• Counter-strategy recommendations (agents, executes, bans)

Tone: Professional, data-driven, actionable.
"""
        else:  # lol
            prompt = f"""
You are a professional League of Legends esports analyst preparing a concise scouting report.

Opponent: {opponent_team}
Matches analyzed: {len(matches_data)}

Raw match data:
{json.dumps(matches_data, indent=2)}

Generate a coach-friendly report (under 300 words) covering:
• Champion pools per role
• Common draft patterns and priority picks/bans
• Early/mid/late game tendencies
• Win conditions
• Weaknesses (exploitable patterns)
• Counter-strategy recommendations (draft, playstyle)

Tone: Professional, data-driven, actionable.
"""

        reasoning.append(ReasoningStep(
            step_number=3,
            description="Generated scouting prompt for LLM"
        ))

        try:
            report = await self.llm.chat(prompt)

            if self._is_invalid_response(report):
                report = f"Scouting report for {opponent_team}: Limited data available. Team shows balanced play with no clear exploitable patterns at this time."
                reasoning.append(ReasoningStep(
                    step_number=5,
                    description="LLM stub/error detected — used fallback report",
                    output_data={"fallback_used": True}
                ))

            # Extracting key insights (simple parsing)
            lines = [line.strip() for line in report.split('\n') if
                     line.strip().startswith(('•', '-', '*', '1.', '2.'))]
            key_insights = lines[:6]

            reasoning.append(ReasoningStep(
                step_number=4,
                description="Scouting report generated successfully",
                output_data={
                    "report_length": len(report),
                    "insights_count": len(key_insights)
                }
            ))

            logger.info("Scouting report completed", extra={"opponent": opponent_team})

            return {
                "opponent": opponent_team,
                "game": game,
                "matches_analyzed": len(matches_data),
                "report": report.strip(),
                "key_insights": key_insights,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Scouting report generation failed", extra={"error": str(e)})
            reasoning.append(ReasoningStep(
                step_number=5,
                description="Scouting report failed",
                output_data={"error": str(e)}
            ))
            return {
                "opponent": opponent_team,
                "error": str(e),
                "reasoning": reasoning,
                "fallback": "Unable to generate report at this time"
            }
