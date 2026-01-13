import json
import logging
from typing import List, Optional

from shared.grid_client import GRIDClient
from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("stats_tracker_agent")


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


class StatsTrackerAgent(MCPAgent):
    def __init__(self):
        super().__init__("StatsTracker")
        self.llm = LLMClient()
        self.grid = GRIDClient()

        self.register_tool("analyze_team_stats", self.analyze_team_stats)
        self.register_tool("get_player_stats", self.get_player_stats)

        logger.info("StatsTrackerAgent initialized")

    def _is_invalid_response(self, response: str) -> bool:
        """Check if LLM response is stub or error"""
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    @log_method
    @metric_counter("stats_tracker")
    async def analyze_team_stats(
            self,
            team_name: str,
            game: str = "valorant",
            recent_matches: int = 10
    ):
        """Main tool: deep analysis of team statistics"""
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Team stats analysis requested",
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

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Fetched recent matches for stats",
            output_data={"matches_count": len(matches_data)}
        ))

        if not matches_data:
            fallback_stats = {
                "win_rate": "50%",
                "avg_rounds_won": 12.5,
                "key_strength": "Balanced play",
                "key_weakness": "Inconsistent clutches"
            }
            return {
                "team": team_name,
                "stats_summary": "Limited data available",
                "fallback_stats": fallback_stats,
                "reasoning": reasoning
            }

        if game == "valorant":
            prompt = f"""
You are a professional Valorant stats analyst.

Team: {team_name}
Analyzed {len(matches_data)} recent matches.

Raw data:
{json.dumps(matches_data, indent=2)}

Provide a concise statistical breakdown:
• Overall win rate
• Map-specific performance (wins/losses per map)
• Average rounds won/lost
• First blood/kill efficiency
• Utility usage patterns
• Clutch win percentage
• Top performing players (by ACS/KD)

Keep under 300 words. Use bullet points. Coach-friendly tone.
"""
        else:  # lol
            prompt = f"""
You are a professional League of Legends stats analyst.

Team: {team_name}
Analyzed {len(matches_data)} recent matches.

Raw data:
{json.dumps(matches_data, indent=2)}

Provide a concise statistical breakdown:
• Overall win rate
• Gold/xp differentials (early/mid/late)
• Objective control (dragons, heralds, barons)
• Vision score and control
• Damage/share per role
• Kill participation
• Top performing players (by KDA, damage)

Keep under 300 words. Use bullet points. Coach-friendly tone.
"""

        try:
            stats_report = await self.llm.chat(prompt)

            if self._is_invalid_response(stats_report):
                stats_report = f"Limited stats available for {team_name}. Win rate ~50%. Balanced performance across roles."
                reasoning.append(ReasoningStep(
                    step_number=4,
                    description="LLM error — used fallback stats report",
                    output_data={"fallback_used": True}
                ))

            reasoning.append(ReasoningStep(
                step_number=3,
                description="Team stats analysis completed",
                output_data={"report_length": len(stats_report)}
            ))

            logger.info("Team stats analysis completed", extra={"team": team_name})

            return {
                "team": team_name,
                "game": game,
                "matches_analyzed": len(matches_data),
                "stats_report": stats_report.strip(),
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Team stats analysis failed", extra={"error": str(e)})
            return {
                "team": team_name,
                "error": str(e),
                "reasoning": reasoning,
                "fallback": "Stats unavailable — check GRID connection"
            }

    @log_method
    @metric_counter("stats_tracker")
    async def get_player_stats(
            self,
            team_name: str,
            player_name: Optional[str] = None,
            game: str = "valorant",
            recent_matches: int = 8
    ):
        """
        Additional tool: detailed statistics of a specific player
        (or the top players of the team, if the player_name is not specified)
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Player stats analysis requested",
            input_data={
                "team": team_name,
                "player": player_name or "team_top_players",
                "game": game,
                "matches": recent_matches
            }
        ))

        matches_data = await self.grid.get_recent_matches(
            team_name=team_name,
            game=game,
            limit=recent_matches
        )

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Fetched recent matches for player stats",
            output_data={"matches_count": len(matches_data)}
        ))

        if not matches_data:
            fallback_stats = {
                "player": player_name or "team_average",
                "avg_kd": "1.1",
                "acs_or_damage": "150 ACS" if game == "valorant" else "25% damage share",
                "key_note": "Solid performer"
            }
            return {
                "team": team_name,
                "player": player_name or "team_average",
                "stats_summary": "Limited data available",
                "fallback_stats": fallback_stats,
                "reasoning": reasoning
            }

        # Creating an prompt
        base_prompt = f"""
You are a professional {game.capitalize()} player performance analyst.

Team: {team_name}
Analyzed {len(matches_data)} recent matches.

Raw match data (with player performances):
{json.dumps(matches_data, indent=2)}
"""

        if player_name:
            specific_prompt = f"""
Focus on player: {player_name}

Extract and summarize their stats:
"""
            if game == "valorant":
                specific_prompt += """
• Average Combat Score (ACS)
• K/D/A ratio
• Headshot percentage
• First bloods / first deaths
• Clutch win rate
• Agent most played and win rate on it
• Strengths / weaknesses
"""
            else:  # lol
                specific_prompt += """
• Average KDA
• CS per minute
• Damage share %
• Vision score
• Gold share
• Kill participation
• Most played champions and win rate
"""
            specific_prompt += "\nKeep under 250 words. Use bullet points. Coach-friendly."
            prompt = base_prompt + specific_prompt
        else:
            team_prompt = """
Identify and rank the top 3 performing players on the team"""
            if game == "valorant":
                team_prompt += " by ACS.\n\nFor each:\n• Player name\n• Average ACS\n• K/D ratio\n• Most played agent\n• Key strength"
            else:
                team_prompt += " by KDA/damage share.\n\nFor each:\n• Player name + role\n• Average KDA\n• Damage or gold share\n• Most played champion\n• Key strength"
            team_prompt += "\n\nAlso provide team average stats.\nKeep under 300 words. Use bullet points."
            prompt = base_prompt + team_prompt

        try:
            player_report = await self.llm.chat(prompt)

            if self._is_invalid_response(player_report):
                player_report = f"Stats for {player_name or 'team players'} limited. Average performance across roles."
                reasoning.append(ReasoningStep(
                    step_number=4,
                    description="LLM error — used fallback player report",
                    output_data={"fallback_used": True}
                ))

            reasoning.append(ReasoningStep(
                step_number=3,
                description="Player stats analysis completed",
                output_data={"report_length": len(player_report)}
            ))

            logger.info("Player stats analysis completed", extra={
                "team": team_name,
                "player": player_name or "team_top"
            })

            return {
                "team": team_name,
                "player": player_name or "team_top_players",
                "game": game,
                "matches_analyzed": len(matches_data),
                "player_report": player_report.strip(),
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Player stats analysis failed", extra={"error": str(e)})
            return {
                "team": team_name,
                "player": player_name or "team_top_players",
                "error": str(e),
                "reasoning": reasoning,
                "fallback": "Player stats unavailable — check GRID connection"
            }
