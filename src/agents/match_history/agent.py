import json
import logging
from typing import List

from shared.grid_client import GRIDClient
from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("match_history_agent")


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


class MatchHistoryAgent(MCPAgent):
    def __init__(self):
        super().__init__("MatchHistory")
        self.llm = LLMClient()
        self.grid = GRIDClient()

        self.register_tool("analyze_match_history", self.analyze_match_history)
        self.register_tool("evaluate_form", self.evaluate_form)

        logger.info("MatchHistoryAgent initialized")

    def _is_invalid_response(self, response: str) -> bool:
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "for more information check", "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    @log_method
    @metric_counter("match_history")
    async def analyze_match_history(
            self,
            team_name: str,
            game: str = "valorant",
            recent_matches: int = 10
    ):
        """
        Match sequence analysis — what happened, what are the trends
        (analogous to commit analysis → game event analysis)
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Match history analysis requested",
            input_data={
                "team": team_name,
                "game": game,
                "matches": recent_matches
            }
        ))

        matches_data = await self.grid.get_recent_matches(
            team_name=team_name,
            game=game,
            limit=recent_matches
        )

        if not matches_data:
            return {
                "team": team_name,
                "error": "No match data available",
                "reasoning": reasoning
            }

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Fetched recent matches",
            output_data={"matches_count": len(matches_data)}
        ))

        # Creating a chronology for LLM
        history_text = ""
        for i, match in enumerate(reversed(matches_data), 1):  # from old to new
            result = "WIN" if match.get("result") == "win" else "LOSS"
            map_name = match.get("map", "Unknown")
            date = match.get("date", "Recent")
            history_text += f"{i}. {date} vs ? — {result} on {map_name}\n"

        prompt = f"""
You are a professional esports analyst tracking team development.

Team: {team_name}
Game: {game.capitalize()}
Last {len(matches_data)} matches (oldest to newest):

{history_text}

Additional match details:
{json.dumps(matches_data, indent=2)}

Summarize their match history:
• Overall record in this period
• Winning/losing streaks
• Performance trends (improving, declining, stable)
• Notable comebacks or collapses
• Adaptation signs (new comps, role swaps, etc.)

Keep under 300 words. Be insightful and coach-friendly.
"""

        try:
            summary = await self.llm.chat(prompt)

            if self._is_invalid_response(summary):
                summary = f"{team_name} has played {len(matches_data)} recent matches with mixed results. Form appears stable."
                reasoning.append(ReasoningStep(
                    step_number=4,
                    description="LLM error — used fallback summary",
                    output_data={"fallback_used": True}
                ))

            reasoning.append(ReasoningStep(
                step_number=3,
                description="Match history summary generated",
                output_data={"summary_length": len(summary)}
            ))

            logger.info("Match history analysis completed", extra={"team": team_name})

            return {
                "team": team_name,
                "game": game,
                "matches_analyzed": len(matches_data),
                "history_summary": summary.strip(),
                "raw_matches": matches_data,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Match history analysis failed", extra={"error": str(e)})
            return {
                "team": team_name,
                "error": str(e),
                "reasoning": reasoning,
                "fallback": "Manual VOD review recommended"
            }

    @log_method
    @metric_counter("match_history")
    async def evaluate_form(
            self,
            team_name: str,
            game: str = "valorant",
            recent_matches: int = 8
    ):
        """
        Current shape estimation (velocity) is an analog of jira_velocity
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Form evaluation requested",
            input_data={"team": team_name, "matches": recent_matches}
        ))

        matches_data = await self.grid.get_recent_matches(
            team_name=team_name,
            game=game,
            limit=recent_matches
        )

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Retrieved matches for form evaluation",
            output_data={"matches_count": len(matches_data)}
        ))

        if not matches_data:
            return {
                "team": team_name,
                "total_matches": 0,
                "win_rate": 0.0,
                "form_status": "no_data",
                "reasoning": reasoning
            }

        wins = sum(1 for m in matches_data if m.get("result") == "win")
        total = len(matches_data)
        win_rate = round((wins / total) * 100, 1)

        # Determining the status of the form
        if win_rate >= 70:
            form_status = "hot_streak"
        elif win_rate >= 55:
            form_status = "good_form"
        elif win_rate >= 40:
            form_status = "inconsistent"
        else:
            form_status = "slumping"

        reasoning.append(ReasoningStep(
            step_number=3,
            description="Form evaluation calculated",
            output_data={"win_rate": win_rate, "form_status": form_status}
        ))

        return {
            "team": team_name,
            "game": game,
            "total_matches": recent_matches,
            "wins": wins,
            "win_rate": win_rate,
            "form_status": form_status,
            "status_description": {
                "hot_streak": "In excellent form — playing at peak",
                "good_form": "Solid and confident play",
                "inconsistent": "Up and down — unpredictable",
                "slumping": "Struggling — need to reset",
                "no_data": "Insufficient data"
            }.get(form_status, ""),
            "reasoning": reasoning
        }
