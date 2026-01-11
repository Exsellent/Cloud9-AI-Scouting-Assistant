import json
import logging
from typing import List

from shared.grid_client import GRIDClient
from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("counter_play_agent")


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


class CounterPlayAgent(MCPAgent):
    def __init__(self):
        super().__init__("CounterPlay")
        self.llm = LLMClient()
        self.grid = GRIDClient()

        self.register_tool("analyze_counter_strategies", self.analyze_counter_strategies)

        logger.info("CounterPlayAgent initialized")

    def _is_invalid_response(self, response: str) -> bool:
        """Check if LLM response is stub or error"""
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "for more information check", "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    @log_method
    @metric_counter("counter_play")
    async def analyze_counter_strategies(
            self,
            opponent_team: str,
            game: str = "valorant",  # "valorant" или "lol"
            recent_matches: int = 8
    ):
        """
        Main tool: analysis of opponent's weaknesses and recommendations on counter-play
        (analogous to risk analysis → exploit detection + mitigation → punishment strategy)
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Counter-play analysis requested",
            input_data={
                "opponent": opponent_team,
                "game": game,
                "matches": recent_matches
            }
        ))

        # Getting match data
        matches_data = await self.grid.get_recent_matches(
            team_name=opponent_team,
            game=game,
            limit=recent_matches
        )

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Fetched opponent recent matches",
            output_data={"matches_found": len(matches_data)}
        ))

        # Creating a prompt for the gamef
        if game == "valorant":
            prompt = f"""
You are an elite Valorant counter-strategy coach with deep analytical skills.

Opponent: {opponent_team}
Analyzed {len(matches_data)} recent matches.

Match data:
{json.dumps(matches_data, indent=2)}

Identify exploitable patterns and generate counter-play recommendations:

1. Weaknesses & Punishable Tendencies
   • Default positions that are predictable
   • Over-relied agents or comps
   • Poor post-plant or retake execution
   • Utility usage mistakes (flashes, smokes)

2. How to Exploit (Counter-Strategies)
   • Recommended agent counters
   • Aggressive executes / rushes that punish their defaults
   • Utility denial or baiting tactics
   • Map control strategies

3. Priority Targets
   • Key players to focus or isolate
   • Roles that carry their rounds

Keep response under 350 words. Use bullet points.
Tone: Aggressive, confident, coach-ready — "This is how we beat them."
"""
        else:  # lol
            prompt = f"""
You are an elite League of Legends counter-strategy analyst.

Opponent: {opponent_team}
Analyzed {len(matches_data)} recent matches.

Match data:
{json.dumps(matches_data, indent=2)}

Identify exploitable patterns and generate counter-strategy:

1. Weaknesses & Exploitable Tendencies
   • Overcommitted jungle pathing
   • Weak early game lanes
   • Poor objective control (dragons/baron)
   • Scaling issues or weak late game

2. Counter-Play Recommendations
   • Priority bans to deny comfort picks
   • Aggressive early invades or ganks
   • Draft counters and team comp advantages
   • Objective priority (what to force)

3. Win Conditions Against Them
   • How to close games quickly
   • How to punish their mistakes

Keep under 350 words. Use bullet points.
Tone: Ruthless, data-driven — "This is their Achilles' heel."
"""

        reasoning.append(ReasoningStep(
            step_number=3,
            description="Generated counter-strategy prompt for LLM"
        ))

        try:
            analysis = await self.llm.chat(prompt)

            if self._is_invalid_response(analysis):
                analysis = f"""Counter-Strategy vs {opponent_team}:

• Limited data available
• Play standard meta with strong early pressure
• Focus on punishing overextensions
• Prioritize objective control"""
                reasoning.append(ReasoningStep(
                    step_number=5,
                    description="LLM error detected — used fallback counter-strategy",
                    output_data={"fallback_used": True}
                ))

            # Extracting key counter-strategies (for a quick overview)
            lines = analysis.split('\n')
            detected_counters = [
                line.strip().lstrip("•-*– ")
                for line in lines
                if line.strip().startswith(('•', '-', '*', '–')) and len(line.strip()) > 10
            ][:8]

            reasoning.append(ReasoningStep(
                step_number=4,
                description="Counter-play analysis completed",
                output_data={
                    "analysis_length": len(analysis),
                    "counter_strategies_count": len(detected_counters)
                }
            ))

            logger.info("Counter-play analysis completed", extra={"opponent": opponent_team})

            return {
                "opponent": opponent_team,
                "game": game,
                "counter_analysis": analysis.strip(),
                "key_counter_strategies": detected_counters,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Counter-play analysis failed", extra={"error": str(e)})
            reasoning.append(ReasoningStep(
                step_number=5,
                description="Counter-play analysis failed",
                output_data={"error": str(e)}
            ))
            return {
                "opponent": opponent_team,
                "error": str(e),
                "reasoning": reasoning,
                "fallback": "Manual VOD review recommended"
            }
