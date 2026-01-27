import json
import logging
from typing import List, Dict, Optional

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
    """
    Match History and Form Analysis Agent (Esports)

    Direct mapping from DevOps Progress Agent:
    - matches
    - analyze_match_history (LLM-powered)
    - evaluate_form (deterministic)

    """

    def __init__(self):
        super().__init__("MatchHistory")
        self.llm = LLMClient()
        self.grid = GRIDClient()

        self.register_tool("analyze_match_history", self.analyze_match_history)
        self.register_tool("evaluate_form", self.evaluate_form)

        logger.info("MatchHistoryAgent initialized with GRID integration")

    def _is_invalid_response(self, response: str) -> bool:
        """Check if LLM response is stub or error"""
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "for more information check", "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    def _next_step(self, reasoning: List[ReasoningStep], description: str,
                   input_data: Optional[Dict] = None, output_data: Optional[Dict] = None):
        """Helper to add sequential reasoning steps"""
        reasoning.append(ReasoningStep(
            step_number=len(reasoning) + 1,
            description=description,
            input_data=input_data or {},
            output_data=output_data or {}
        ))

    @log_method
    @metric_counter("match_history")
    async def analyze_match_history(
            self,
            team_name: str,
            game: str = "valorant",
            recent_matches: int = 10
    ):
        """
        Match sequence analysis
        game event analysis
        LLM-powered with proper fallback
        """
        reasoning: List[ReasoningStep] = []
        fallback_used = False

        # Step 1: Request received
        self._next_step(reasoning, "Match history analysis requested",
                        input_data={
                            "team": team_name,
                            "game": game,
                            "matches": recent_matches
                        })

        # Step 2: Fetch match data from GRID
        matches_data = await self.grid.get_recent_matches(
            team_name=team_name,
            game=game,
            limit=recent_matches
        )

        if not matches_data:
            self._next_step(reasoning, "No match data available from GRID",
                            output_data={"matches_found": 0})
            return {
                "team": team_name,
                "game": game,
                "matches_analyzed": 0,
                "error": "No match data available",
                "reasoning": reasoning
            }

        self._next_step(reasoning, "Fetched recent matches from GRID",
                        output_data={"matches_count": len(matches_data)})

        # Create chronology for LLM (oldest to newest)
        history_text = ""
        for i, match in enumerate(reversed(matches_data), 1):
            result = "WIN" if match.get("result") == "win" else "LOSS"
            map_name = match.get("map", "Unknown")
            opponent = match.get("opponent", "?")
            date = match.get("date", "Recent")
            history_text += f"{i}. {date} vs {opponent} — {result} on {map_name}\n"

        # Generate LLM prompt
        prompt = f"""
You are a professional esports analyst tracking team development.

Team: {team_name}
Game: {game.capitalize()}
Last {len(matches_data)} matches (oldest to newest):

{history_text}

Match details:
{json.dumps(matches_data, indent=2)}

Provide a comprehensive match history summary:

• Overall record in this period (W-L)
• Winning/losing streaks and momentum shifts
• Performance trends (improving, declining, stable)
• Notable comebacks, upsets, or collapses
• Adaptation signs (new compositions, role swaps, strategy changes)
• Map pool strengths and weaknesses

Keep under 300 words. Be insightful, data-driven, and coach-friendly.
Tone: Professional analyst preparing scouting report.
"""

        try:
            # Attempt LLM analysis
            summary = await self.llm.chat(prompt)

            # Check if LLM response is valid
            if self._is_invalid_response(summary):
                fallback_used = True
                wins = sum(1 for m in matches_data if m.get("result") == "win")
                losses = len(matches_data) - wins
                summary = (
                    f"{team_name} Match History Summary:\n\n"
                    f"Recent record: {wins}W - {losses}L over last {len(matches_data)} matches.\n"
                    f"Form appears stable. Manual VOD review recommended for detailed insights."
                )
                logger.warning("Match History Agent using fallback summary",
                               extra={"team": team_name, "matches": len(matches_data)})

            # Step 3: Analysis completed
            self._next_step(reasoning, "Match history summary generated",
                            output_data={
                                "summary_length": len(summary),
                                "fallback_used": fallback_used
                            })

            # Step 4 (optional): Fallback annotation
            if fallback_used:
                self._next_step(reasoning, "Fallback summary used due to LLM unavailability",
                                output_data={"matches_analyzed": len(matches_data)})

            logger.info("Match history analysis completed",
                        extra={"team": team_name, "matches": len(matches_data), "fallback": fallback_used})

            return {
                "team": team_name,
                "game": game,
                "matches_analyzed": len(matches_data),
                "history_summary": summary.strip(),
                "fallback_used": fallback_used,
                "raw_matches": matches_data,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Match history analysis failed", extra={"error": str(e)})

            # Use fallback even on exception
            wins = sum(1 for m in matches_data if m.get("result") == "win")
            losses = len(matches_data) - wins

            self._next_step(reasoning, "Match history analysis failed with exception — using fallback",
                            output_data={"error": str(e), "fallback_used": True})

            self._next_step(reasoning, "Fallback summary generated",
                            output_data={"matches_analyzed": len(matches_data)})

            return {
                "team": team_name,
                "game": game,
                "matches_analyzed": len(matches_data),
                "history_summary": f"{team_name}: {wins}W-{losses}L. Analysis failed. Manual VOD review recommended.",
                "fallback_used": True,
                "reasoning": reasoning,
                "error": str(e)
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
        Current form evaluation

        DETERMINISTIC agent - no LLM required
        Demonstrates universal architecture for both AI and traditional logic
        """
        reasoning: List[ReasoningStep] = []

        # Step 1: Request received
        self._next_step(reasoning, "Form evaluation requested",
                        input_data={
                            "team": team_name,
                            "game": game,
                            "matches": recent_matches
                        })

        # Step 2: Fetch matches from GRID
        matches_data = await self.grid.get_recent_matches(
            team_name=team_name,
            game=game,
            limit=recent_matches
        )

        self._next_step(reasoning, "Retrieved matches for form evaluation",
                        output_data={"matches_count": len(matches_data)})

        # Early exit if no data
        if not matches_data:
            self._next_step(reasoning, "No matches found — returning zero metrics")

            return {
                "team": team_name,
                "game": game,
                "total_matches": 0,
                "win_rate": 0.0,
                "form_status": "no_data",
                "reasoning": reasoning
            }

        # Step 3: Calculate form metrics (deterministic logic)
        wins = sum(1 for m in matches_data if m.get("result") == "win")
        total = len(matches_data)
        win_rate = round((wins / total) * 100, 1)

        # Form status classification (4 levels like velocity_status)
        if win_rate >= 70:
            form_status = "hot_streak"
            description = "In excellent form — playing at peak level"
        elif win_rate >= 55:
            form_status = "good_form"
            description = "Solid and confident play"
        elif win_rate >= 40:
            form_status = "inconsistent"
            description = "Up and down — unpredictable results"
        else:
            form_status = "slumping"
            description = "Struggling — need to reset and refocus"

        self._next_step(reasoning, "Form evaluation calculated",
                        output_data={
                            "win_rate": win_rate,
                            "form_status": form_status,
                            "wins": wins,
                            "total": total
                        })

        logger.info("Form evaluation completed",
                    extra={
                        "team": team_name,
                        "win_rate": win_rate,
                        "form_status": form_status
                    })

        return {
            "team": team_name,
            "game": game,
            "total_matches": total,
            "matches_analyzed": recent_matches,
            "wins": wins,
            "losses": total - wins,
            "win_rate": win_rate,
            "form_status": form_status,
            "status_description": description,
            "reasoning": reasoning
        }
