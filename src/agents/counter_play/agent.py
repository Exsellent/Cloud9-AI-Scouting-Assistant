import json
import logging
from typing import List, Dict, Optional

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
    """
    Counter-Strategy and Weakness Analysis Agent (Esports):

    - exploitable weaknesses
    - counter-play tactics
    - opponent weakness detection

    """

    # Baseline counter-strategies when LLM is unavailable
    BASELINE_VALORANT_COUNTERS = [
        "Apply early aggression to test opponent defaults",
        "Focus utility denial and trading kills efficiently",
        "Exploit map control weaknesses in mid rounds",
        "Target key fraggers with coordinated trades",
        "Maintain disciplined post-plant positioning"
    ]

    BASELINE_LOL_COUNTERS = [
        "Prioritize early game jungle pressure",
        "Deny vision control around key objectives",
        "Target scaling carries with early ganks",
        "Force objective contests during power spikes",
        "Maintain wave control to deny CS leads"
    ]

    def __init__(self):
        super().__init__("CounterPlay")
        self.llm = LLMClient()
        self.grid = GRIDClient()

        self.register_tool("analyze_counter_strategies", self.analyze_counter_strategies)

        logger.info("CounterPlayAgent initialized with baseline counter-strategies")

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

    def _get_baseline_counters(self, game: str) -> List[str]:
        """Get conservative baseline counter-strategies based on game"""
        if game.lower() == "valorant":
            return self.BASELINE_VALORANT_COUNTERS
        elif game.lower() == "lol":
            return self.BASELINE_LOL_COUNTERS
        else:
            return [
                "Maintain disciplined standard play",
                "Focus on fundamentals and positioning",
                "Capitalize on opponent mistakes",
                "Control tempo and map pressure"
            ]

    @log_method
    @metric_counter("counter_play")
    async def analyze_counter_strategies(
            self,
            opponent_team: str,
            game: str = "valorant",
            recent_matches: int = 8
    ):
        """
        Counter-strategy analysis — opponent weaknesses and exploitation tactics
        LLM-powered with intelligent fallback to baseline tactics
        """
        reasoning: List[ReasoningStep] = []
        fallback_used = False

        # Step 1: Request received
        self._next_step(reasoning, "Counter-play analysis requested",
                        input_data={
                            "opponent": opponent_team,
                            "game": game,
                            "matches": recent_matches
                        })

        # Step 2: Fetch opponent match data from GRID
        matches_data = await self.grid.get_recent_matches(
            team_name=opponent_team,
            game=game,
            limit=recent_matches
        )

        self._next_step(reasoning, "Fetched opponent recent matches from GRID",
                        output_data={"matches_found": len(matches_data)})

        # Generate game-specific prompt
        if game.lower() == "valorant":
            prompt = f"""
You are an elite Valorant counter-strategy coach with deep analytical skills.

Opponent: {opponent_team}
Analyzed {len(matches_data)} recent matches.

Match data:
{json.dumps(matches_data, indent=2)}

Identify exploitable patterns and generate counter-play recommendations:

1. **Weaknesses & Punishable Tendencies**
   • Default positions that are predictable
   • Over-relied agents or compositions
   • Poor post-plant or retake execution
   • Utility usage mistakes (flashes, smokes, walls)
   • Eco round vulnerabilities

2. **How to Exploit (Counter-Strategies)**
   • Recommended agent counters
   • Aggressive executes/rushes that punish their defaults
   • Utility denial or baiting tactics
   • Map control strategies to force rotations
   • Anti-stratting their known setups

3. **Priority Targets**
   • Key players to focus or isolate
   • Roles that carry their rounds
   • Players who tilt under pressure

Format: Use bullet points starting with '•' for each strategy.
Keep response under 350 words.
Tone: Aggressive, confident, coach-ready — "This is how we beat them."
"""
        else:  # League of Legends
            prompt = f"""
You are an elite League of Legends counter-strategy analyst.

Opponent: {opponent_team}
Analyzed {len(matches_data)} recent matches.

Match data:
{json.dumps(matches_data, indent=2)}

Identify exploitable patterns and generate counter-strategy:

1. **Weaknesses & Exploitable Tendencies**
   • Overcommitted jungle pathing patterns
   • Weak early game lanes or matchups
   • Poor objective control (dragons/baron timing)
   • Scaling issues or weak power spikes
   • Vision control gaps

2. **Counter-Play Recommendations**
   • Priority bans to deny comfort picks
   • Aggressive early invades or ganks
   • Draft counters and team comp advantages
   • Objective priority (what to force, what to trade)
   • Split push or teamfight decisions

3. **Win Conditions Against Them**
   • How to close games quickly (fast tempo)
   • How to punish their scaling or mistakes
   • Mid-game power spike exploitation

Format: Use bullet points starting with '•' for each strategy.
Keep under 350 words.
Tone: Ruthless, data-driven — "This is their Achilles' heel."
"""

        try:
            # Attempt LLM analysis
            analysis = await self.llm.chat(prompt)

            # Check if LLM response is valid
            if self._is_invalid_response(analysis):
                fallback_used = True
                baseline_counters = self._get_baseline_counters(game)

                analysis = (
                        f"⚠️ LLM analysis unavailable. Applied baseline counter-strategies.\n\n"
                        f"Counter-Strategy vs {opponent_team}:\n\n" +
                        "\n".join(f"• {counter}" for counter in baseline_counters)
                )

                detected_counters = baseline_counters

                logger.warning("CounterPlay Agent using baseline strategies",
                               extra={"opponent": opponent_team, "game": game})
            else:
                # Extract counter-strategies from analysis
                lines = analysis.split('\n')
                detected_counters = [
                    line.strip().lstrip("•-*— ")
                    for line in lines
                    if line.strip().startswith(('•', '-', '*', '—')) and len(line.strip()) > 10
                ][:10]  # Top 10 strategies

                # Fallback if parsing failed
                if not detected_counters:
                    fallback_used = True
                    detected_counters = self._get_baseline_counters(game)
                    analysis += f"\n\n⚠️ Supplemented with baseline counters:\n" + \
                                "\n".join(f"• {counter}" for counter in detected_counters)

            # Step 3: Analysis completed
            self._next_step(reasoning, "Counter-play analysis completed",
                            output_data={
                                "analysis_length": len(analysis),
                                "counter_strategies_count": len(detected_counters),
                                "fallback_used": fallback_used
                            })

            # Step 4 (optional): Fallback annotation
            if fallback_used:
                self._next_step(reasoning, "Baseline counter-strategies applied due to LLM unavailability",
                                output_data={"baseline_strategies": len(detected_counters)})

            logger.info("Counter-play analysis completed",
                        extra={
                            "opponent": opponent_team,
                            "game": game,
                            "strategies": len(detected_counters),
                            "fallback": fallback_used
                        })

            return {
                "opponent": opponent_team,
                "game": game,
                "matches_analyzed": len(matches_data),
                "counter_analysis": analysis.strip(),
                "key_counter_strategies": detected_counters,
                "fallback_used": fallback_used,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Counter-play analysis failed", extra={"error": str(e)})

            # Use baseline counters even on exception
            baseline_counters = self._get_baseline_counters(game)

            self._next_step(reasoning, "Counter-play analysis failed with exception — using baseline",
                            output_data={"error": str(e), "fallback_used": True})

            self._next_step(reasoning, "Baseline counter-strategies applied",
                            output_data={"strategies_count": len(baseline_counters)})

            return {
                "opponent": opponent_team,
                "game": game,
                "matches_analyzed": len(matches_data),
                "counter_analysis": (
                        f"⚠️ Analysis failed: {str(e)}\n\n"
                        f"Baseline Counter-Strategies vs {opponent_team}:\n" +
                        "\n".join(f"• {counter}" for counter in baseline_counters)
                ),
                "key_counter_strategies": baseline_counters,
                "fallback_used": True,
                "reasoning": reasoning,
                "error": str(e)
            }
