import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from shared.grid_client import GRIDClient
from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("draft_coach_agent")


@dataclass
class ChampionStats:
    """Structured champion statistics"""
    champion: str
    games: int
    winrate: float
    comfort_level: str  # "high", "medium", "low"
    threat_level: str  # "must_ban", "consider", "low"


@dataclass
class DraftPhase:
    """Draft phase information"""
    phase: str  # "ban_1", "pick_1", "ban_2", "pick_2", etc.
    our_bans: List[str]
    their_bans: List[str]
    our_picks: List[str]
    their_picks: List[str]


def log_method(func):
    """Decorator for logging method calls"""

    async def wrapper(self, *args, **kwargs):
        logger.info(f"{func.__name__} called")
        try:
            result = await func(self, *args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {str(e)}")
            raise

    return wrapper


class DraftCoachAgent(MCPAgent):
    """
    Professional LoL Draft Coach Agent with Advanced Analytics

    Key Features:
    1. Data-Driven Champion Pool Analysis (GRID integration)
    2. Phase-Aware Draft Recommendations
    3. Deterministic + LLM Hybrid Approach
    4. Confidence Scoring & Explainability

    Proper sequential reasoning, no reasoning.extend(), explicit fallbacks
    """

    # Deterministic thresholds for high-priority threats
    COMFORT_THRESHOLD_GAMES = 5
    MUST_BAN_WINRATE = 0.65
    HIGH_PRIORITY_GAMES = 7

    def __init__(self):
        super().__init__("DraftCoach")
        self.llm = LLMClient()
        self.grid = GRIDClient()

        self.register_tool("recommend_draft", self.recommend_draft)
        self.register_tool("analyze_opponent_pool", self.analyze_opponent_pool)
        self.register_tool("suggest_ban_priority", self.suggest_ban_priority)
        self.register_tool("evaluate_draft_state", self.evaluate_draft_state)

        logger.info("DraftCoachAgent initialized with advanced analytics")

    def _next_step(self, reasoning: List[ReasoningStep], description: str,
                   input_data: Optional[Dict] = None, output_data: Optional[Dict] = None):
        """ Helper to add sequential reasoning steps"""
        reasoning.append(ReasoningStep(
            step_number=len(reasoning) + 1,
            description=description,
            input_data=input_data or {},
            output_data=output_data or {}
        ))

    def _is_invalid_response(self, response: str) -> bool:
        """Check if LLM response is stub or error"""
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    def _extract_high_risk_champions(
            self,
            champion_pools: Dict[str, List[Dict]]
    ) -> List[Dict[str, Any]]:
        """
        Deterministic threat detection (hybrid approach)

        Identifies must-ban champions based on hard statistics
        """
        threats = []

        for role, champions in champion_pools.items():
            for champ in champions:
                games = champ.get("games", 0)
                winrate = float(champ.get("winrate", "0%").rstrip("%")) / 100

                threat_level = "low"
                reason = []

                # Deterministic rules
                if winrate >= self.MUST_BAN_WINRATE:
                    threat_level = "must_ban"
                    reason.append(f"{winrate * 100:.0f}% winrate")

                if games >= self.HIGH_PRIORITY_GAMES:
                    if threat_level == "low":
                        threat_level = "consider"
                    reason.append(f"{games} games (high comfort)")

                if threat_level != "low":
                    threats.append({
                        "champion": champ.get("champion"),
                        "role": role,
                        "games": games,
                        "winrate": winrate,
                        "threat_level": threat_level,
                        "reason": ", ".join(reason)
                    })

        # Sort by threat level, then winrate
        threats.sort(
            key=lambda x: (
                0 if x["threat_level"] == "must_ban" else 1,
                -x["winrate"]
            )
        )

        return threats

    def _calculate_confidence(
            self,
            matches_count: int,
            data_source: str,
            fallback_used: bool
    ) -> float:
        """
        Calculate confidence score for recommendations

        Based on:
        - Data availability
        - Fallback usage
        - Sample size
        """
        if fallback_used:
            return 0.3

        # Base confidence from sample size
        if matches_count >= 10:
            base_confidence = 0.85
        elif matches_count >= 5:
            base_confidence = 0.65
        else:
            base_confidence = 0.45

        # Bonus for real data source
        if data_source == "GRID":
            base_confidence += 0.10

        return min(base_confidence, 0.95)

    def _safe_parse_json(
            self,
            response: str,
            default: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Safely parse JSON from LLM response

        Returns None if parsing fails (for explicit fallback detection)
        """
        # Try direct JSON parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Extract from ```json block
        import re
        json_block = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1))
            except json.JSONDecodeError:
                pass

        # Find balanced braces
        start = response.find('{')
        if start == -1:
            logger.warning("No JSON found in LLM response")
            return default

        brace_count = 0
        end = start
        for i in range(start, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        if brace_count == 0:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse JSON from LLM response")
        return default

    # ========================================================================
    # MAIN TOOLS
    # ========================================================================

    @log_method
    @metric_counter("draft_coach")
    async def analyze_opponent_pool(
            self,
            opponent_team: str,
            recent_matches: int = 10
    ) -> Dict[str, Any]:
        """
        Opponent's champion pool analysis (LoL only)

        Returns structured data (dict), not string
        Proper sequential reasoning with explicit fallbacks
        """
        reasoning: List[ReasoningStep] = []
        fallback_used = False

        # Step 1: Request received
        self._next_step(reasoning, "Opponent champion pool analysis requested",
                        input_data={"opponent": opponent_team, "matches": recent_matches})

        # Step 2: Fetch data from GRID
        matches_data = []
        try:
            matches_data = await self.grid.get_recent_matches(
                team_name=opponent_team,
                game="lol",
                limit=recent_matches
            )

            self._next_step(reasoning, "Fetched opponent match data from GRID",
                            output_data={
                                "matches_found": len(matches_data),
                                "data_source": "GRID"
                            })

        except Exception as e:
            logger.error("GRID data fetch failed", extra={"error": str(e)})

            # Explicit fallback step
            self._next_step(reasoning, "GRID data fetch failed - will use mock data",
                            output_data={"error": str(e)})

        # Step 3: LLM analysis
        champion_pools = None

        if matches_data:
            prompt = f"""
You are a professional LoL draft analyst.
Analyze the last {len(matches_data)} matches of {opponent_team} and extract champion pools per role.

Raw data:
{json.dumps(matches_data, indent=2)[:2000]}

Return ONLY valid JSON (no markdown, no backticks):
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
                champion_pools = self._safe_parse_json(response, None)

                # Check if parsing succeeded
                if champion_pools is None or self._is_invalid_response(response):
                    fallback_used = True

                    # Explicit fallback step
                    self._next_step(reasoning, "LLM response invalid - using baseline champion pools",
                                    output_data={"fallback_used": True})
                else:
                    self._next_step(reasoning, "Champion pools analyzed from LLM",
                                    output_data={
                                        "roles_analyzed": len(champion_pools),
                                        "fallback_used": False
                                    })

            except Exception as e:
                logger.error("LLM analysis failed", extra={"error": str(e)})
                fallback_used = True

                # Explicit exception fallback step
                self._next_step(reasoning, "LLM analysis failed - using baseline champion pools",
                                output_data={"error": str(e), "fallback_used": True})

        # Fallback champion pools (structured data, not string)
        if fallback_used or not champion_pools:
            champion_pools = {
                "top": [{"champion": "KSante", "games": 6, "winrate": "67%"}],
                "jungle": [{"champion": "Vi", "games": 5, "winrate": "80%"}],
                "mid": [{"champion": "Azir", "games": 7, "winrate": "71%"}],
                "adc": [{"champion": "Kalista", "games": 4, "winrate": "75%"}],
                "support": [{"champion": "Rell", "games": 8, "winrate": "62%"}]
            }

        # Step 4 - Deterministic threat detection
        detected_threats = self._extract_high_risk_champions(champion_pools)

        self._next_step(reasoning, "Deterministic threat detection completed",
                        output_data={
                            "must_ban_champions": len([t for t in detected_threats if t["threat_level"] == "must_ban"]),
                            "total_threats": len(detected_threats)
                        })

        # Step 5 - Calculate confidence
        confidence = self._calculate_confidence(
            matches_count=len(matches_data),
            data_source="GRID" if matches_data else "fallback",
            fallback_used=fallback_used
        )

        self._next_step(reasoning, "Champion pool analysis completed",
                        output_data={
                            "confidence_level": confidence,
                            "total_champions_analyzed": sum(len(v) for v in champion_pools.values())
                        })

        logger.info("Opponent pool analysis completed",
                    extra={
                        "opponent": opponent_team,
                        "matches": len(matches_data),
                        "confidence": confidence
                    })

        return {
            "opponent": opponent_team,
            "champion_pools": champion_pools,
            "detected_threats": detected_threats,
            "confidence_level": confidence,
            "matches_analyzed": len(matches_data),
            "fallback_used": fallback_used,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("draft_coach")
    async def suggest_ban_priority(
            self,
            opponent_team: str,
            our_side: str = "blue",
            banned_champions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
       Suggest ban priority based on opponent pool analysis

        Combines deterministic rules + LLM strategic reasoning
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(reasoning, "Ban priority suggestion requested",
                        input_data={"opponent": opponent_team, "side": our_side})

        # Step 2: Analyze opponent pool
        pool_result = await self.analyze_opponent_pool(opponent_team)

        # Proper composition - reference, not extend
        self._next_step(reasoning, "Opponent pool analysis completed",
                        output_data={
                            "detected_threats": len(pool_result.get("detected_threats", [])),
                            "confidence": pool_result.get("confidence_level", 0.5),
                            "analysis_steps": len(pool_result.get("reasoning", []))
                        })

        # Step 3: Generate ban recommendations
        detected_threats = pool_result.get("detected_threats", [])
        banned_champions = banned_champions or []

        # Filter out already banned champions
        available_threats = [
            t for t in detected_threats
            if t["champion"] not in banned_champions
        ]

        # Deterministic top 3 bans
        priority_bans = available_threats[:3]

        self._next_step(reasoning, "Generated ban priority list",
                        output_data={
                            "priority_bans": len(priority_bans),
                            "already_banned": len(banned_champions)
                        })

        # Completion step
        self._next_step(reasoning, "Ban priority suggestion completed",
                        output_data={
                            "total_steps": len(reasoning),
                            "confidence": pool_result.get("confidence_level", 0.5)
                        })

        return {
            "opponent": opponent_team,
            "priority_bans": priority_bans,
            "detected_threats": detected_threats,
            "confidence_level": pool_result.get("confidence_level", 0.5),
            "pool_analysis": pool_result.get("reasoning", []),  # Separate reference
            "reasoning": reasoning
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
            opponent_pool_summary: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Main draft recommendation tool (League of Legends only)

        Proper sequential reasoning with phase awareness
        """
        if game.lower() != "lol":
            return {
                "error": "DraftCoach supports only League of Legends drafting.",
                "recommendation": "For Valorant agent selection, use CounterPlay agent.",
                "reasoning": [ReasoningStep(
                    step_number=1,
                    description="Invalid game for draft - LoL only"
                )]
            }

        reasoning: List[ReasoningStep] = []

        # Step 1: Draft recommendation requested
        self._next_step(reasoning, "Draft recommendation requested",
                        input_data={
                            "opponent": opponent_team,
                            "our_side": our_side,
                            "game": game,
                            "current_phase": current_phase
                        })

        # Step 2: Opponent pool analysis (if not provided)
        pool_confidence = 0.5
        detected_threats = []

        if not opponent_pool_summary:
            pool_result = await self.analyze_opponent_pool(opponent_team)
            opponent_pool_summary = pool_result.get("champion_pools", {})
            detected_threats = pool_result.get("detected_threats", [])
            pool_confidence = pool_result.get("confidence_level", 0.5)

            # Proper composition - reference, not extend
            self._next_step(reasoning, "Opponent pool analysis completed",
                            output_data={
                                "matches_analyzed": pool_result.get("matches_analyzed", 0),
                                "detected_threats": len(detected_threats),
                                "confidence": pool_confidence,
                                "analysis_steps": len(pool_result.get("reasoning", []))
                            })
        else:
            # Pool summary provided externally
            self._next_step(reasoning, "Using provided opponent pool summary",
                            output_data={"source": "external"})

        #  Step 3 - Phase-aware prompt construction
        phase_context = ""
        if current_phase:
            phase_context = f"""
CURRENT DRAFT PHASE: {current_phase.upper()}
Focus recommendations ONLY for this specific phase.
"""

        # Construct deterministic threat summary
        threat_summary = ""
        if detected_threats:
            must_bans = [t for t in detected_threats if t["threat_level"] == "must_ban"]
            if must_bans:
                threat_summary = "\n**STATISTICAL MUST-BANS:**\n" + "\n".join([
                    f"- {t['champion']} ({t['role']}): {t['reason']}"
                    for t in must_bans[:3]
                ])

        prompt = f"""
You are an expert League of Legends draft coach for a professional team.

Our side: {our_side.upper()}
Opponent: {opponent_team}
{phase_context}

Opponent champion pools (from recent games):
{json.dumps(opponent_pool_summary, indent=2)[:1500]}
{threat_summary}

Current patch strong/meta champions: KSante, Maokai, Azir, Orianna, Kalista, Rell, Sejuani, Vi.

Banned champions: {banned_champions or "None yet"}
Picked champions: {picked_champions or "None yet"}

Task:
1. Recommend 3-5 priority bans targeting opponent's comfort/high-winrate champions
2. Recommend strong picks with synergies for our team
3. Explain key counters and win conditions

Response format (strict):
## Priority Bans
- Champion1 (reason)
- Champion2 (reason)

## Recommended Picks
- Role: Champion (reason)

## Key Synergies/Counters
...

## Win Conditions
...

Keep under 400 words. Professional, confident, coach-friendly tone.
"""

        # Step 4: Generate recommendations
        recommendation = None
        recommendation_fallback = False

        try:
            recommendation = await self.llm.chat(prompt)

            if self._is_invalid_response(recommendation):
                recommendation_fallback = True

                # Explicit fallback step
                self._next_step(reasoning, "LLM recommendation invalid - using baseline draft",
                                output_data={"fallback_used": True})
            else:
                self._next_step(reasoning, "Draft recommendations generated from LLM",
                                output_data={
                                    "response_length": len(recommendation),
                                    "fallback_used": False
                                })

        except Exception as e:
            logger.error("Draft recommendation failed", extra={"error": str(e)})
            recommendation_fallback = True

            # Explicit exception fallback step
            self._next_step(reasoning, "LLM recommendation failed - using baseline draft",
                            output_data={"error": str(e), "fallback_used": True})

        # Fallback recommendation
        if recommendation_fallback or not recommendation:
            recommendation = f"""## Priority Bans
- Target opponent's top comfort picks from pool analysis
{threat_summary if threat_summary else "- Focus on meta-defining champions"}

## Recommended Picks
- Top: KSante/Maokai (strong frontline)
- Jungle: Vi/Sejuani (engage + control)
- Mid: Azir/Orianna (scaling + utility)
- ADC: Kalista (objective control)
- Support: Rell (engage)

## Key Synergies/Counters
Strong frontline with engage capabilities. Prioritize objective control.

## Win Conditions
Early pressure and objective control. Scale to teamfight advantage.
"""

        # Step 5 - Calculate overall confidence
        overall_confidence = pool_confidence
        if recommendation_fallback:
            overall_confidence *= 0.5

        # Step 6 - Generate key assumptions for explainability
        key_assumptions = []
        if detected_threats:
            top_threat = detected_threats[0]
            key_assumptions.append(f"Opponent relies on {top_threat['champion']} ({top_threat['role']})")
        if our_side == "blue":
            key_assumptions.append("Blue side first pick advantage utilized")
        else:
            key_assumptions.append("Red side counter-pick advantage prioritized")

        # Step 7 - Draft recommendation completed
        self._next_step(reasoning, "Draft recommendation finalized",
                        output_data={
                            "confidence_level": overall_confidence,
                            "phase_aware": bool(current_phase),
                            "key_assumptions": key_assumptions,
                            "fallback_used": recommendation_fallback
                        })

        logger.info("Draft recommendation completed",
                    extra={
                        "opponent": opponent_team,
                        "confidence": overall_confidence,
                        "phase": current_phase
                    })

        return {
            "opponent": opponent_team,
            "our_side": our_side,
            "game": game,
            "current_phase": current_phase,
            "recommendation": recommendation.strip(),
            "detected_threats": detected_threats,
            "confidence_level": overall_confidence,
            "key_assumptions": key_assumptions,  # Explainability
            "fallback_used": recommendation_fallback,
            "pool_analysis": opponent_pool_summary,  # Structured data
            "reasoning": reasoning  # Main reasoning only
        }

    @log_method
    @metric_counter("draft_coach")
    async def evaluate_draft_state(
            self,
            our_team_comp: List[str],
            their_team_comp: List[str],
            remaining_picks: int
    ) -> Dict[str, Any]:
        """
        Evaluate current draft state and suggest adjustments

        Advanced feature for real-time draft coaching
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(reasoning, "Draft state evaluation requested",
                        input_data={
                            "our_picks": len(our_team_comp),
                            "their_picks": len(their_team_comp),
                            "remaining": remaining_picks
                        })

        prompt = f"""
Evaluate this League of Legends draft state:

OUR TEAM: {', '.join(our_team_comp)}
THEIR TEAM: {', '.join(their_team_comp)}
REMAINING PICKS: {remaining_picks}

Provide:
1. Current draft advantage (our favor / their favor / neutral)
2. Missing roles/strengths in our comp
3. Recommended final picks
4. Win conditions for our comp

Keep analysis under 300 words.
"""

        try:
            evaluation = await self.llm.chat(prompt)

            self._next_step(reasoning, "Draft state evaluation completed",
                            output_data={"evaluation_generated": True})

        except Exception as e:
            evaluation = "Draft evaluation unavailable - manual review recommended"

            self._next_step(reasoning, "Draft state evaluation failed",
                            output_data={"error": str(e)})

        self._next_step(reasoning, "Draft state evaluation finalized",
                        output_data={"total_steps": len(reasoning)})

        return {
            "our_team_comp": our_team_comp,
            "their_team_comp": their_team_comp,
            "evaluation": evaluation,
            "reasoning": reasoning
        }


# ========================================================================
# DATACLASS HELPERS (for type hints)
# ========================================================================

def champion_stats_from_dict(data: Dict) -> ChampionStats:
    """Convert dict to ChampionStats dataclass"""
    winrate_str = data.get("winrate", "0%")
    winrate = float(winrate_str.rstrip("%")) / 100

    games = data.get("games", 0)

    # Determine comfort level
    if games >= 7:
        comfort = "high"
    elif games >= 4:
        comfort = "medium"
    else:
        comfort = "low"

    # Determine threat level
    if winrate >= 0.65:
        threat = "must_ban"
    elif games >= 7 or winrate >= 0.55:
        threat = "consider"
    else:
        threat = "low"

    return ChampionStats(
        champion=data.get("champion", "Unknown"),
        games=games,
        winrate=winrate,
        comfort_level=comfort,
        threat_level=threat
    )
