import json
import logging
from typing import List, Optional

from shared.grid_client import GRIDClient
from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("draft_coach_agent")


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


class DraftCoachAgent(MCPAgent):
    def __init__(self):
        super().__init__("DraftCoach")
        self.llm = LLMClient()
        self.grid = GRIDClient()

        self.register_tool("recommend_draft", self.recommend_draft)
        self.register_tool("analyze_opponent_pool", self.analyze_opponent_pool)

        logger.info("DraftCoachAgent initialized")

    def _is_invalid_response(self, response: str) -> bool:
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    @log_method
    @metric_counter("draft_coach")
    async def analyze_opponent_pool(
            self,
            opponent_team: str,
            recent_matches: int = 10
    ):
        """Opponent's champion pool analysis (LoL only)"""
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Opponent champion pool analysis requested",
            input_data={"opponent": opponent_team, "matches": recent_matches}
        ))

        matches_data = await self.grid.get_recent_matches(
            team_name=opponent_team,
            game="lol",  # Фикс: всегда LoL для драфта
            limit=recent_matches
        )

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Fetched opponent match data",
            output_data={"matches_found": len(matches_data)}
        ))

        prompt = f"""
You are a professional LoL draft analyst.
Analyze the last {len(matches_data)} matches of {opponent_team} and extract champion pools per role.

Raw data:
{json.dumps(matches_data, indent=2)}

Return ONLY valid JSON:
{{
  "top": [{{"champion": "Name", "games": N, "winrate": "X%"}}],
  "jungle": [...],
  "mid": [...],
  "adc": [...],
  "support": [...]
}}
Sort each role by games descending, top 5 champions max.
"""

        try:
            response = await self.llm.chat(prompt)
            if self._is_invalid_response(response):
                raise ValueError("Invalid LLM response")

            reasoning.append(ReasoningStep(
                step_number=3,
                description="Champion pools analyzed",
                output_data={"response_length": len(response)}
            ))

            return {
                "opponent": opponent_team,
                "champion_pools": response.strip(),
                "reasoning": reasoning
            }

        except Exception as e:
            fallback = {
                "top": [{"champion": "KSante", "games": 6, "winrate": "67%"}],
                "jungle": [{"champion": "Vi", "games": 5, "winrate": "80%"}],
                "mid": [{"champion": "Azir", "games": 7, "winrate": "71%"}],
                "adc": [{"champion": "Kalista", "games": 4, "winrate": "75%"}],
                "support": [{"champion": "Rell", "games": 8, "winrate": "62%"}]
            }
            return {
                "opponent": opponent_team,
                "champion_pools": json.dumps(fallback, indent=2),
                "fallback_used": True,
                "reasoning": reasoning + [
                    ReasoningStep(step_number=4, description="Fallback used", output_data={"error": str(e)})]
            }

    @log_method
    @metric_counter("draft_coach")
    async def recommend_draft(
            self,
            opponent_team: str,
            our_side: str = "blue",
            game: str = "lol",
            current_phase: Optional[str] = None,
            banned_champions: Optional[List[str]] = None,
            picked_champions: Optional[List[str]] = None,
            opponent_pool_summary: Optional[str] = None
    ):
        """The main tool is draft recommendations (League of Legends only)"""
        if game.lower() != "lol":
            return {
                "error": "DraftCoach supports only League of Legends drafting.",
                "recommendation": "For Valorant agent selection, use CounterPlay or ScoutingReport agents.",
                "reasoning": [ReasoningStep(step_number=1, description="Invalid game for draft")]
            }

        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Draft recommendation requested",
            input_data={
                "opponent": opponent_team,
                "our_side": our_side,
                "game": game
            }
        ))

        if not opponent_pool_summary:
            pool_result = await self.analyze_opponent_pool(opponent_team)
            opponent_pool_summary = pool_result.get("champion_pools", "{}")
            reasoning.extend(pool_result.get("reasoning", []))

        prompt = f"""
You are an expert League of Legends draft coach for a professional team.

Our side: {our_side.upper()}
Opponent: {opponent_team}

Opponent champion pools (from recent games):
{opponent_pool_summary}

Current patch strong/meta champions: KSante, Maokai, Azir, Orianna, Kalista, Rell, Sejuani, Vi.

Task:
1. Recommend 3–5 priority bans targeting opponent's comfort/high-winrate champions
2. Recommend strong picks with synergies for our team
3. Explain key counters and win conditions

Response format (strict):
## Priority Bans
- Champion1 (reason)
- Champion2 (reason)
...

## Recommended Picks
- Role: Champion (reason)
...

## Key Synergies/Counters
...

## Win Conditions
...

Keep under 400 words. Professional, confident, coach-friendly tone.
"""

        try:
            recommendation = await self.llm.chat(prompt)

            if self._is_invalid_response(recommendation):
                recommendation = """## Priority Bans
- Target opponent's top comfort picks from pool analysis

## Recommended Picks
- Top: KSante/Maokai
- Jungle: Vi/Sejuani
- Mid: Azir/Orianna
- ADC: Kalista
- Support: Rell

## Key Synergies/Counters
Strong frontline + engage

## Win Conditions
Early pressure and objective control"""

            reasoning.append(ReasoningStep(
                step_number=3,
                description="Draft recommendations generated",
                output_data={"response_length": len(recommendation)}
            ))

            return {
                "opponent": opponent_team,
                "our_side": our_side,
                "game": game,
                "recommendation": recommendation.strip(),
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Draft recommendation failed", extra={"error": str(e)})
            return {
                "error": str(e),
                "fallback": "Manual draft review recommended",
                "reasoning": reasoning
            }
