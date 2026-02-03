import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from shared.grid_client import GRIDClient
from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.models import ReasoningStep

logger = logging.getLogger("draft_coach_agent")


# ========================================================================
# Domain Safety with Normalized Whitelist
# ========================================================================

# Basic normalization function
def normalize_champion_name(name: str) -> str:
    """Normalize champion name for comparison"""
    return name.lower().strip().replace("'", "").replace(" ", "")


# The original names
_LOL_CHAMPION_NAMES = {
    # Top laners
    "aatrox", "akali", "camille", "cho'gath", "darius", "fiora", "gangplank",
    "garen", "gnar", "gragas", "gwen", "illaoi", "irelia", "jax", "jayce",
    "kayle", "kennen", "ksante", "malphite", "maokai", "mordekaiser", "nasus",
    "olaf", "ornn", "poppy", "quinn", "renekton", "riven", "rumble", "sett",
    "shen", "sion", "sylas", "tahm kench", "teemo", "trundle", "tryndamere",
    "urgot", "vayne", "vladimir", "volibear", "warwick", "wukong", "yasuo", "yone",

    # Junglers
    "amumu", "bel'veth", "brand", "briar", "diana", "ekko", "elise", "evelynn",
    "fiddlesticks", "graves", "hecarim", "ivern", "jarvan iv", "karthus",
    "kayn", "kha'zix", "kindred", "lee sin", "lillia", "master yi", "nidalee",
    "nocturne", "nunu", "rammus", "rek'sai", "rengar", "sejuani", "shaco",
    "shyvana", "skarner", "taliyah", "udyr", "vi", "viego", "xin zhao", "zac",

    # Mid laners
    "ahri", "akshan", "anivia", "annie", "aurelion sol", "azir", "cassiopeia",
    "corki", "fizz", "galio", "heimerdinger", "hwei", "kassadin", "katarina",
    "leblanc", "lissandra", "lux", "malzahar", "naafiri", "neeko", "orianna",
    "pantheon", "qiyana", "ryze", "syndra", "talon", "twisted fate", "veigar",
    "vel'koz", "vex", "viktor", "vladimir", "xerath", "yasuo", "yone",
    "zed", "ziggs", "zoe",

    # ADC
    "aphelios", "ashe", "caitlyn", "draven", "ezreal", "jhin", "jinx",
    "kai'sa", "kalista", "kog'maw", "lucian", "miss fortune", "nilah",
    "samira", "sivir", "smolder", "tristana", "twitch", "varus", "vayne",
    "xayah", "zeri",

    # Supports
    "alistar", "bard", "blitzcrank", "braum", "janna", "karma", "leona",
    "lulu", "lux", "milio", "morgana", "nami", "nautilus", "pyke", "rakan",
    "rell", "renata glasc", "senna", "seraphine", "sona", "soraka", "taric",
    "thresh", "yuumi", "zilean", "zyra"
}

# Normalized whitelist without duplicates
LOL_CHAMPIONS = {normalize_champion_name(name) for name in _LOL_CHAMPION_NAMES}


def is_valid_lol_champion(name: str) -> bool:
    """Check if champion name is valid for LoL"""
    return normalize_champion_name(name) in LOL_CHAMPIONS


# ========================================================================
# DATACLASSES
# ========================================================================

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
    5. Domain Safety (LoL champion validation)

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

        logger.info("DraftCoachAgent initialized with advanced analytics and domain safety")

    def _next_step(self, reasoning: List[ReasoningStep], description: str,
                   input_data: Optional[Dict] = None, output_data: Optional[Dict] = None):
        """Helper to add sequential reasoning steps"""
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

    def _contains_invalid_champions(self, text: str) -> List[str]:
        """
        Detect invalid (non-LoL) champions in free-text LLM output
        Explicitly detects Valorant agents and other invalid names
        """
        # Known Valorant agents to explicitly detect
        VALORANT_AGENTS = {
            'jett', 'phoenix', 'sage', 'sova', 'viper', 'cypher', 'brimstone', 'omen',
            'killjoy', 'breach', 'raze', 'reyna', 'skye', 'yoru', 'astra', 'kayo', 'kay/o',
            'chamber', 'neon', 'fade', 'harbor', 'gekko', 'deadlock', 'iso', 'clove', 'vyse'
        }

        invalid = []

        # First pass: Check for explicit Valorant agents (case-insensitive)
        text_lower = text.lower()
        for agent in VALORANT_AGENTS:
            # Use word boundary regex to avoid partial matches
            pattern = r'\b' + re.escape(agent) + r'\b'
            if re.search(pattern, text_lower):
                # Find the actual capitalized version in the text
                actual_match = re.search(r'\b' + re.escape(agent) + r'\b', text, re.IGNORECASE)
                if actual_match:
                    invalid.append(actual_match.group(0))

        # Second pass: Extract potential champion names (capitalized words, but not possessives)
        # Exclude words ending with 's (possessive) and common sentence starters
        words = re.findall(r"\b[A-Z][A-Za-z'/\-]+\b", text)

        # Common words to skip (massively expanded list)
        COMMON_WORDS = {
            # Common sentence starters and transitions
            'the', 'our', 'their', 'we', 'this', 'that', 'these', 'those', 'it', 'its',
            'for', 'with', 'from', 'into', 'during', 'including', 'until', 'against',
            'among', 'throughout', 'despite', 'towards', 'upon', 'concerning', 'by',
            'use', 'using', 'given', 'since', 'while', 'when', 'where', 'which',
            'early', 'mid', 'late', 'focus', 'after', 'before', 'between',

            # Draft-specific terms
            'priority', 'recommended', 'key', 'win', 'conditions', 'synergies',
            'counters', 'team', 'strong', 'good', 'high', 'low', 'top', 'jungle',
            'support', 'adc', 'bans', 'picks', 'composition', 'potential', 'meta',

            # Team names
            'sentinels', 'blue', 'red', 'side', 'draft', 'game', 'match', 'league',
            'legends', 'champions',

            # Gameplay terms
            'crowd', 'control', 'damage', 'mobility', 'protection', 'winrate',
            'games', 'versatile', 'aggressive', 'defensive', 'roaming', 'vision',
            'map', 'objectives', 'towers', 'drakes', 'fights', 'pace', 'plays',
            'engage', 'peel', 'teamfight', 'split', 'push', 'wave', 'clear',

            # Other game references
            'agent', 'agents', 'valorant', 'overwatch', 'dota'
        }

        for word in words:
            # Skip possessive forms (e.g., "Elise's", "Tristana's")
            if word.endswith("'s") or word.endswith("'s"):
                continue

            # Skip common words
            if word.lower() in COMMON_WORDS:
                continue

            # Skip if already found as Valorant agent
            if any(word.lower() == inv.lower() for inv in invalid):
                continue

            # Check if it's a valid LoL champion
            if not is_valid_lol_champion(word):
                invalid.append(word)

        return list(set(invalid))

    def _filter_invalid_champions_from_text(self, text: str, invalid_champions: List[str]) -> str:
        """
        Remove invalid champion names from text and replace with warning
        More aggressive filtering: removes entire lines containing invalid champions
        """
        if not invalid_champions:
            return text

        lines = text.split('\n')
        filtered_lines = []
        removed_lines = []

        for line in lines:
            # Check if this line contains any invalid champion
            contains_invalid = False
            for invalid_champ in invalid_champions:
                # Case-insensitive search
                if re.search(rf'\b{re.escape(invalid_champ)}\b', line, re.IGNORECASE):
                    contains_invalid = True
                    removed_lines.append(line.strip())
                    break

            if not contains_invalid:
                filtered_lines.append(line)

        filtered_text = '\n'.join(filtered_lines)

        # Add detailed warning
        warning = (
            f"\n\n{'=' * 60}\n"
            f"⚠️  **DOMAIN VALIDATION WARNING** ⚠️\n"
            f"{'=' * 60}\n"
            f"Invalid champions detected and removed: {', '.join(sorted(set(invalid_champions)))}\n\n"
            f"**Reason**: Only League of Legends champions are valid for this game.\n"
            f"Valorant agents and other game characters are not applicable.\n\n"
            f"**Lines removed**: {len(removed_lines)}\n"
            f"{'=' * 60}\n"
        )

        return filtered_text + warning

    def _generate_safe_fallback_recommendation(
            self,
            opponent: str,
            our_side: str,
            champion_pools: Dict[str, List[Dict]]
    ) -> str:
        """
        Generate a safe fallback recommendation using only validated champion data
        This is used when LLM generates invalid responses with non-LoL champions
        """
        recommendation_parts = [
            "## ⚠️ Safe Draft Recommendation (Data-Driven Fallback)\n",
            "*Note: This recommendation is generated purely from validated champion data.*\n"
        ]

        # Priority Bans based on high threat champions
        recommendation_parts.append("\n### Priority Bans\n")
        high_threat_champs = []

        for role, champions in champion_pools.items():
            for champ in champions[:3]:  # Top 3 per role
                winrate = champ.get('winrate', 0)
                games = champ.get('games', 0)

                if winrate > 60 or games > 7:
                    threat_reason = []
                    if winrate > 65:
                        threat_reason.append("very high winrate")
                    if games > 10:
                        threat_reason.append("extensive experience")
                    elif games > 7:
                        threat_reason.append("high comfort level")

                    reason_text = " and ".join(threat_reason) if threat_reason else "consistent performance"

                    high_threat_champs.append({
                        'text': f"- **{champ['champion']}** ({role}): {games} games, {winrate:.1f}% WR - {reason_text}",
                        'priority': winrate * games  # Composite score
                    })

        # Sort by priority and take top 5
        high_threat_champs.sort(key=lambda x: x['priority'], reverse=True)

        if high_threat_champs:
            for champ_info in high_threat_champs[:5]:
                recommendation_parts.append(champ_info['text'])
        else:
            recommendation_parts.append("- Target their most-played champions based on match data\n")

        # Meta picks by role
        recommendation_parts.append("\n### Recommended Meta Picks by Role\n")
        recommendation_parts.append(
            "Based on current League of Legends meta:\n"
            "- **Top**: K'Sante, Aatrox, Camille, Jax (strong teamfight and split-push)\n"
            "- **Jungle**: Sejuani, Vi, Jarvan IV, Lee Sin (engage and objective control)\n"
            "- **Mid**: Azir, Orianna, Sylas, Ahri (wave clear and teamfight impact)\n"
            "- **ADC**: Jinx, Caitlyn, Kai'Sa, Ashe (consistent damage output)\n"
            "- **Support**: Thresh, Nautilus, Lulu, Rell (crowd control and protection)\n"
        )

        # General strategy
        recommendation_parts.append("\n### Strategic Approach\n")
        recommendation_parts.append(
            f"1. **Ban Priority**: Focus on their highest winrate and most-played champions\n"
            f"2. **Side Advantage**: Utilize {our_side} side advantages for counter-picks\n"
            f"3. **Team Composition**: Build around engage, peel, and damage balance\n"
            f"4. **Win Conditions**: \n"
            f"   - Early game: Secure jungle priority and vision control\n"
            f"   - Mid game: Contest objectives (dragons, towers)\n"
            f"   - Late game: Coordinate teamfights around key cooldowns\n"
        )

        recommendation_parts.append(
            f"\n### Data Summary\n"
            f"- Opponent: {opponent}\n"
            f"- Roles analyzed: {len(champion_pools)}\n"
            f"- Total unique champions in pool: {sum(len(champs) for champs in champion_pools.values())}\n"
        )

        return "\n".join(recommendation_parts)

    # ========================================================================
    #  Domain Validation
    # ========================================================================

    def _validate_champion_pools(
            self,
            champion_pools: Dict[str, List[Dict]],
            reasoning: List[ReasoningStep]
    ) -> tuple[Dict[str, List[Dict]], bool]:
        """
        Validate that champion pools contain only valid LoL champions

        Returns:
            (validated_pools, domain_violation_detected)
        """
        domain_violation = False
        validated_pools = {}
        invalid_champions = []

        for role, champions in champion_pools.items():
            validated_champions = []
            for champ in champions:
                champ_name = champ.get("champion", "")

                if is_valid_lol_champion(champ_name):
                    validated_champions.append(champ)
                else:
                    domain_violation = True
                    invalid_champions.append(champ_name)
                    logger.warning(f"Invalid LoL champion detected in {role}: {champ_name}")

            validated_pools[role] = validated_champions

        if domain_violation:
            self._next_step(
                reasoning,
                "Domain validation failed - invalid champions removed",
                output_data={
                    "invalid_champions": invalid_champions,
                    "violation_count": len(invalid_champions)
                }
            )

        return validated_pools, domain_violation

    def _validate_and_filter_recommendation(
            self,
            recommendation: str,
            reasoning: List[ReasoningStep],
            opponent: str,
            our_side: str,
            champion_pools: Dict[str, List[Dict]]
    ) -> str:
        """
        Validate final recommendation text and filter out invalid champions

        Returns:
            Cleaned recommendation text or safe fallback if too many violations
        """
        invalid_champions = self._contains_invalid_champions(recommendation)

        if not invalid_champions:
            self._next_step(
                reasoning,
                "Final recommendation validated - no invalid champions found",
                output_data={"validation_passed": True}
            )
            return recommendation

        # Log validation failure
        logger.error(f"Invalid champions found in recommendation: {invalid_champions}")
        self._next_step(
            reasoning,
            "Final recommendation validation FAILED",
            output_data={
                "invalid_champions": invalid_champions,
                "violation_count": len(invalid_champions),
                "action": "filtering_invalid_champions"
            }
        )

        # If too many violations (> 3), use safe fallback
        if len(invalid_champions) > 3:
            logger.warning("Too many invalid champions - using safe fallback")
            self._next_step(
                reasoning,
                "Using safe fallback recommendation due to excessive violations",
                output_data={"fallback_reason": "too_many_invalid_champions"}
            )
            return self._generate_safe_fallback_recommendation(opponent, our_side, champion_pools)

        # Filter invalid champions from text
        filtered_recommendation = self._filter_invalid_champions_from_text(
            recommendation,
            invalid_champions
        )

        self._next_step(
            reasoning,
            "Invalid champions filtered from recommendation",
            output_data={
                "filtered_champions": invalid_champions,
                "filtered_text_length": len(filtered_recommendation)
            }
        )

        return filtered_recommendation

    # ========================================================================
    #  Tool: Analyze Opponent Pool (deterministic + LLM summary)
    # ========================================================================

    @log_method
    async def analyze_opponent_pool(
            self,
            opponent: str,
            matches: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze opponent champion pool from GRID data

        Steps:
        1. Fetch raw match data
        2. Aggregate by role + champion
        3. Validate champions (domain safety)
        4. Generate LLM summary
        5. Check for hallucinations

        Returns:
            {
                "champion_pools": {role: [champions]},
                "priority_threats": [],
                "summary": str,
                "reasoning": []
            }
        """
        reasoning: List[ReasoningStep] = []
        self._next_step(
            reasoning,
            "Opponent champion pool analysis requested",
            input_data={"opponent": opponent, "matches": matches}
        )

        # Step 1: Fetch match data
        try:
            recent_matches = await self.grid.get_team_matches(opponent, limit=matches)
            self._next_step(
                reasoning,
                "Fetched opponent match data",
                output_data={"matches_found": len(recent_matches)}
            )
        except Exception as e:
            self._next_step(
                reasoning,
                "Failed to fetch match data",
                output_data={"error": str(e)}
            )
            return {
                "champion_pools": {},
                "priority_threats": [],
                "summary": "Unable to analyze opponent - data unavailable",
                "reasoning": reasoning
            }

        # Step 2: Aggregate by role
        role_pools: Dict[str, Dict[str, Dict]] = {
            "top": {}, "jungle": {}, "mid": {}, "adc": {}, "support": {}
        }

        for match in recent_matches:
            for player in match.get("players", []):
                role = player.get("role", "").lower()
                champion = player.get("champion", "")

                if role not in role_pools or not champion:
                    continue

                if champion not in role_pools[role]:
                    role_pools[role][champion] = {"games": 0, "wins": 0}

                role_pools[role][champion]["games"] += 1
                if player.get("win", False):
                    role_pools[role][champion]["wins"] += 1

        # Convert to sorted list format
        champion_pools = {}
        for role, champs in role_pools.items():
            pool_list = []
            for champ_name, stats in champs.items():
                games = stats["games"]
                winrate = (stats["wins"] / games * 100) if games > 0 else 0
                pool_list.append({
                    "champion": champ_name,
                    "games": games,
                    "winrate": winrate
                })
            # Sort by games desc, then winrate desc
            pool_list.sort(key=lambda x: (x["games"], x["winrate"]), reverse=True)
            champion_pools[role] = pool_list

        self._next_step(
            reasoning,
            "Champion pools aggregated by role",
            output_data={"roles_analyzed": len(champion_pools)}
        )

        # Step 3: Validate champions (domain safety)
        validated_pools, domain_violation = self._validate_champion_pools(
            champion_pools, reasoning
        )

        if domain_violation:
            champion_pools = validated_pools

        # Step 4: LLM summary with strict prompt
        pool_summary_lines = []
        for role, champs in champion_pools.items():
            if champs:
                top3 = champs[:3]
                champ_strs = [
                    f"{c['champion']} ({c['games']}g, {c['winrate']:.1f}% WR)"
                    for c in top3
                ]
                pool_summary_lines.append(f"{role.upper()}: {', '.join(champ_strs)}")

        pool_text = "\n".join(pool_summary_lines)

        prompt = f"""You are a professional League of Legends draft analyst.

Analyze this opponent champion pool data and provide insights:

{pool_text}

**CRITICAL RULES:**
1. ONLY mention champions that appear in the data above
2. Use ONLY League of Legends champion names
3. DO NOT mention Valorant agents (Jett, KAY/O, Omen, Sova, etc.)
4. Focus on win rates, play frequency, and role comfort
5. Keep response under 200 words

Provide:
1. Top 3 priority threats (champions with high WR or games)
2. Role-specific comfort picks
3. Potential ban targets

Use ONLY the champions listed in the data above."""

        try:
            summary = await self.llm.chat(prompt)
            self._next_step(
                reasoning,
                "Champion pools analyzed",
                output_data={"response_length": len(summary)}
            )

            # Step 5: Check for hallucinations
            if self._is_invalid_response(summary):
                summary = "Analysis unavailable - using deterministic data only"
                self._next_step(
                    reasoning,
                    "LLM response invalid - fallback to deterministic",
                    output_data={"fallback_used": True}
                )

            # NEW: Check for invalid champions in summary
            invalid_champs = self._contains_invalid_champions(summary)
            if invalid_champs:
                logger.error(f"Invalid champions in summary: {invalid_champs}")
                summary = self._filter_invalid_champions_from_text(summary, invalid_champs)
                self._next_step(
                    reasoning,
                    "Invalid champions detected and filtered from summary",
                    output_data={
                        "invalid_champions": invalid_champs,
                        "filtered": True
                    }
                )

        except Exception as e:
            summary = "Analysis error - review data manually"
            self._next_step(
                reasoning,
                "LLM analysis failed",
                output_data={"error": str(e)}
            )

        # Deterministic priority threats
        all_threats = []
        for role, champs in champion_pools.items():
            for c in champs:
                if c["winrate"] >= self.MUST_BAN_WINRATE * 100 or c["games"] >= self.HIGH_PRIORITY_GAMES:
                    all_threats.append(c["champion"])

        self._next_step(
            reasoning,
            "Priority threats identified",
            output_data={"threat_count": len(all_threats)}
        )

        return {
            "champion_pools": champion_pools,
            "priority_threats": all_threats[:5],
            "summary": summary,
            "reasoning": reasoning
        }

    # ========================================================================
    #  Tool: Recommend Draft
    # ========================================================================

    @log_method
    async def recommend_draft(
            self,
            opponent_team: str,
            our_side: str = "blue",
            game: str = "lol"
    ) -> Dict[str, Any]:
        """
        Generate draft recommendations

        Steps:
        1. Analyze opponent pool (calls analyze_opponent_pool)
        2. LLM generates recommendation based on validated data
        3. Validate final recommendation text
        4. Filter or fallback if invalid champions found

        Returns:
            {
                "opponent": str,
                "our_side": str,
                "game": str,
                "recommendation": str,
                "reasoning": []
            }
        """
        reasoning: List[ReasoningStep] = []
        self._next_step(
            reasoning,
            "Draft recommendation requested",
            input_data={"opponent": opponent_team, "our_side": our_side, "game": game}
        )

        # Step 1: Analyze opponent (nested tool call)
        pool_result = await self.analyze_opponent_pool(opponent_team)
        champion_pools = pool_result["champion_pools"]
        priority_threats = pool_result["priority_threats"]

        # Merge reasoning chains
        reasoning.extend(pool_result["reasoning"])

        # Step 2: Build context for LLM
        pool_context_lines = []
        for role, champs in champion_pools.items():
            if champs:
                top3 = champs[:3]
                champ_strs = [
                    f"{c['champion']} ({c['games']}g, {c['winrate']:.1f}% WR)"
                    for c in top3
                ]
                pool_context_lines.append(f"**{role.upper()}**: {', '.join(champ_strs)}")

        pool_context = "\n".join(pool_context_lines)

        prompt = f"""You are a professional League of Legends draft coach.

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**
1. This is a LEAGUE OF LEGENDS match - NOT Valorant, NOT any other game
2. ONLY use League of Legends champion names (e.g., Aatrox, Sejuani, Azir, Jinx, Thresh)
3. NEVER EVER use Valorant agent names (Jett, Omen, KAY/O, Sova, Sage, Phoenix, etc.)
4. NEVER EVER use Overwatch hero names (Tracer, Genji, etc.)
5. NEVER EVER use Dota 2 hero names
6. ALL recommendations must use only LoL champions from the provided data

**INCORRECT EXAMPLE (DO NOT DO THIS):**
- "Ban Jett" ❌ (Jett is from Valorant)
- "Pick Sova" ❌ (Sova is from Valorant)

**CORRECT EXAMPLE:**
- "Ban Sejuani" ✓ (League of Legends champion)
- "Pick Azir" ✓ (League of Legends champion)

**Opponent**: {opponent_team}
**Our Side**: {our_side}
**Game**: League of Legends

**Opponent Champion Pool Data (LEAGUE OF LEGENDS ONLY)**:
{pool_context}

**Priority Threats**: {', '.join(priority_threats) if priority_threats else 'None identified'}

Provide a complete draft strategy with:
1. **Priority Bans** (3-5 champions from their pool data - MUST be LoL champions)
2. **Recommended Picks** (suggest meta League of Legends champions for each role: Top, Jungle, Mid, ADC, Support)
3. **Key Synergies/Counters** (using LoL mechanics and champions)
4. **Win Conditions** (based on LoL gameplay)

Keep under 400 words. 
REMINDER: Use ONLY League of Legends champions. Double-check every champion name before writing."""

        try:
            recommendation = await self.llm.chat(prompt)
            self._next_step(
                reasoning,
                "Draft recommendations generated",
                output_data={"response_length": len(recommendation)}
            )

            # Step 3 & 4: Validate and filter final recommendation
            recommendation = self._validate_and_filter_recommendation(
                recommendation=recommendation,
                reasoning=reasoning,
                opponent=opponent_team,
                our_side=our_side,
                champion_pools=champion_pools
            )

        except Exception as e:
            logger.error(f"Draft recommendation failed: {str(e)}")
            self._next_step(
                reasoning,
                "Draft recommendation generation failed",
                output_data={"error": str(e)}
            )
            recommendation = self._generate_safe_fallback_recommendation(
                opponent_team, our_side, champion_pools
            )

        return {
            "opponent": opponent_team,
            "our_side": our_side,
            "game": game,
            "recommendation": recommendation,
            "reasoning": reasoning
        }

    # ========================================================================
    #  Tool: Suggest Ban Priority
    # ========================================================================

    @log_method
    async def suggest_ban_priority(
            self,
            opponent: str,
            available_bans: int = 5
    ) -> Dict[str, Any]:
        """
        Suggest ban priority list based on opponent data

        Deterministic approach:
        1. Must-ban: WR >= 65% or games >= 10
        2. High-priority: WR >= 55% or games >= 7
        3. Consider: Rest

        Returns:
            {
                "ban_list": [champions],
                "reasoning": [],
                "confidence": float
            }
        """
        reasoning: List[ReasoningStep] = []
        self._next_step(
            reasoning,
            "Ban priority calculation requested",
            input_data={"opponent": opponent, "available_bans": available_bans}
        )

        pool_result = await self.analyze_opponent_pool(opponent)
        champion_pools = pool_result["champion_pools"]

        # Aggregate all champions across roles
        all_champions = []
        for role, champs in champion_pools.items():
            for c in champs:
                champ_stat = champion_stats_from_dict(c)
                all_champions.append(champ_stat)

        # Sort by threat level
        must_ban = [c for c in all_champions if c.threat_level == "must_ban"]
        consider = [c for c in all_champions if c.threat_level == "consider"]

        # Sort each tier by games desc
        must_ban.sort(key=lambda x: x.games, reverse=True)
        consider.sort(key=lambda x: x.games, reverse=True)

        ban_list = []
        for c in must_ban[:available_bans]:
            ban_list.append(c.champion)

        remaining = available_bans - len(ban_list)
        if remaining > 0:
            for c in consider[:remaining]:
                ban_list.append(c.champion)

        self._next_step(
            reasoning,
            "Ban priority list generated",
            output_data={"ban_count": len(ban_list)}
        )

        # Confidence based on data quality
        total_games = sum(c.games for c in all_champions)
        confidence = min(1.0, total_games / 50.0)

        self._next_step(
            reasoning,
            "Confidence calculated",
            output_data={"confidence": confidence, "total_games": total_games}
        )

        return {
            "ban_list": ban_list,
            "reasoning": reasoning,
            "confidence": confidence
        }

    # ========================================================================
    #  Tool: Evaluate Draft State
    # ========================================================================

    @log_method
    async def evaluate_draft_state(
            self,
            our_picks: List[str],
            their_picks: List[str],
            our_bans: List[str] = None,
            their_bans: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate current draft state and provide insights

        Args:
            our_picks: List of our champion picks
            their_picks: List of enemy champion picks
            our_bans: List of our bans
            their_bans: List of enemy bans

        Returns:
            {
                "our_team_comp": str,
                "their_team_comp": str,
                "evaluation": str,
                "reasoning": []
            }
        """
        reasoning: List[ReasoningStep] = []
        our_bans = our_bans or []
        their_bans = their_bans or []

        self._next_step(
            reasoning,
            "Draft state evaluation requested",
            input_data={
                "our_picks": our_picks,
                "their_picks": their_picks,
                "our_bans": our_bans,
                "their_bans": their_bans
            }
        )

        # Validate all champions
        all_champions = our_picks + their_picks + our_bans + their_bans
        invalid_champions = []

        for champ in all_champions:
            if not is_valid_lol_champion(champ):
                invalid_champions.append(champ)
                logger.warning(f"Invalid champion in draft state: {champ}")

        if invalid_champions:
            self._next_step(
                reasoning,
                "Invalid champions detected in draft state",
                output_data={"invalid_champions": invalid_champions}
            )

        our_team_comp = ", ".join(our_picks) if our_picks else "None"
        their_team_comp = ", ".join(their_picks) if their_picks else "None"

        prompt = f"""You are a professional League of Legends draft analyst.

**CRITICAL INSTRUCTIONS:**
1. ONLY analyze League of Legends champions
2. DO NOT mention Valorant agents or other game characters
3. Focus on LoL-specific strategies, synergies, and win conditions

**Current Draft State**:
- Our Team: {our_team_comp}
- Their Team: {their_team_comp}
- Our Bans: {', '.join(our_bans) if our_bans else 'None'}
- Their Bans: {', '.join(their_bans) if their_bans else 'None'}

Provide analysis of:
1. Team composition strengths/weaknesses
2. Key matchups to watch
3. Power spikes (early/mid/late game)
4. Win conditions for our comp

Keep analysis under 300 words.
Use ONLY valid League of Legends champions in recommendations."""

        try:
            evaluation = await self.llm.chat(prompt)
            self._next_step(
                reasoning,
                "Draft state evaluation completed",
                output_data={"evaluation_generated": True}
            )

            # Validate and filter evaluation text
            invalid_champs_in_eval = self._contains_invalid_champions(evaluation)
            if invalid_champs_in_eval:
                logger.error(f"Invalid champions in evaluation: {invalid_champs_in_eval}")
                evaluation = self._filter_invalid_champions_from_text(
                    evaluation,
                    invalid_champs_in_eval
                )
                self._next_step(
                    reasoning,
                    "Invalid champions filtered from evaluation",
                    output_data={"filtered_champions": invalid_champs_in_eval}
                )

        except Exception as e:
            evaluation = "Draft evaluation unavailable - manual review recommended"
            self._next_step(
                reasoning,
                "Draft state evaluation failed",
                output_data={"error": str(e)}
            )

        self._next_step(
            reasoning,
            "Draft state evaluation finalized",
            output_data={"total_steps": len(reasoning)}
        )

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
    """
    Convert dict to ChampionStats dataclass
    Standalone utility function, NOT part of DraftCoachAgent class

    Usage:
        stats = champion_stats_from_dict(champ_data)
    """
    winrate_str = data.get("winrate", "0%") if isinstance(data.get("winrate"), str) else f"{data.get('winrate', 0)}%"
    winrate = float(str(winrate_str).rstrip("%")) / 100
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


# ========================================================================
# Optional Reasoning Step Unification Helper
# ========================================================================

def unify_reasoning_chains(
        parent_reasoning: List[ReasoningStep],
        child_reasoning: List[ReasoningStep],
        parent_step_id: str = "main"
) -> List[ReasoningStep]:
    """
    Optional helper to unify nested reasoning chains for linear trace

    Usage (optional - only if you need unified trace):
        unified = unify_reasoning_chains(main_reasoning, pool_reasoning, "draft")

    Features:
    - Adds scope prefix to step descriptions
    - Maintains original step ordering
    - Optional for backward compatibility

    Note: Agent currently uses scoped reasoning (each tool has own chain)
    """
    unified = []

    for step in parent_reasoning:
        unified.append(ReasoningStep(
            step_number=len(unified) + 1,
            description=f"[{parent_step_id}] {step.description}",
            input_data=step.input_data,
            output_data=step.output_data
        ))

    for step in child_reasoning:
        unified.append(ReasoningStep(
            step_number=len(unified) + 1,
            description=f"[pool] {step.description}",
            input_data=step.input_data,
            output_data=step.output_data
        ))

    return unified
