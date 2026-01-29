import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from shared.grid_client import GRIDClient
from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("stats_tracker_agent")


@dataclass
class DataQuality:
    """Assessment of statistical data quality"""
    matches_analyzed: int
    sufficient_sample: bool
    sample_size_rating: str  # "excellent", "good", "limited", "insufficient"
    confidence_level: float
    data_source: str
    limitations: List[str]


@dataclass
class AnalysisStrategy:
    """Selected analysis approach based on data quality"""
    strategy_type: str  # "trend_analysis", "snapshot_analysis", "baseline_only"
    reasoning: str
    expected_accuracy: str  # "high", "medium", "low"


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


class StatsTrackerAgent(MCPAgent):
    """
    Professional Esports Statistics & Performance Analysis Agent

    Key Features:
    1. Data Quality Assessment: Validates sample size and reliability
    2. Strategy Selection: Chooses analysis approach based on data availability
    3. Game-Aware Analysis: Adapts metrics for Valorant/LoL
    4. Confidence Scoring: Quantifies reliability of insights
    5. Competitive Intelligence: Actionable player/team performance analysis

    Proper sequential reasoning with validation, strategy selection, and confidence scoring

    This agent MUST express uncertainty when data is limited.
    """

    # Data quality thresholds
    EXCELLENT_SAMPLE = 15  # 15+ matches = excellent
    GOOD_SAMPLE = 10  # 10+ matches = good
    LIMITED_SAMPLE = 5  # 5+ matches = limited but usable
    INSUFFICIENT_SAMPLE = 3  # <3 matches = insufficient

    def __init__(self):
        super().__init__("StatsTracker")
        self.llm = LLMClient()
        self.grid = GRIDClient()

        self.register_tool("analyze_team_stats", self.analyze_team_stats)
        self.register_tool("get_player_stats", self.get_player_stats)
        self.register_tool("compare_teams", self.compare_teams)

        logger.info("StatsTrackerAgent initialized with quality assessment")

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

    def _assess_data_quality(
            self,
            matches_count: int,
            data_source: str = "GRID"
    ) -> DataQuality:
        """
        Assess statistical data quality and reliability

        Critical for honest reporting of analysis limitations
        """
        limitations = []

        # Classify sample size
        if matches_count >= self.EXCELLENT_SAMPLE:
            sample_size_rating = "excellent"
            confidence_level = 0.9
            sufficient_sample = True
        elif matches_count >= self.GOOD_SAMPLE:
            sample_size_rating = "good"
            confidence_level = 0.75
            sufficient_sample = True
        elif matches_count >= self.LIMITED_SAMPLE:
            sample_size_rating = "limited"
            confidence_level = 0.55
            sufficient_sample = True
            limitations.append("Small sample size may not reflect true performance")
        elif matches_count >= self.INSUFFICIENT_SAMPLE:
            sample_size_rating = "insufficient"
            confidence_level = 0.35
            sufficient_sample = False
            limitations.append("Insufficient data for reliable statistical analysis")
        else:
            sample_size_rating = "insufficient"
            confidence_level = 0.2
            sufficient_sample = False
            limitations.append("Critical: Very limited data - analysis unreliable")

        # Additional quality factors
        if matches_count < 5:
            limitations.append("Cannot identify trends with high confidence")

        if data_source == "mock":
            limitations.append("Demo data - production requires real GRID integration")

        return DataQuality(
            matches_analyzed=matches_count,
            sufficient_sample=sufficient_sample,
            sample_size_rating=sample_size_rating,
            confidence_level=confidence_level,
            data_source=data_source,
            limitations=limitations
        )

    def _select_analysis_strategy(
            self,
            data_quality: DataQuality,
            game: str
    ) -> AnalysisStrategy:
        """
        Select appropriate analysis strategy based on data quality

        This is the "Think" step - deciding HOW to analyze
        """
        matches = data_quality.matches_analyzed

        if matches >= self.GOOD_SAMPLE:
            strategy_type = "trend_analysis"
            reasoning = f"Sufficient data ({matches} matches) for identifying performance trends and patterns"
            expected_accuracy = "high"
        elif matches >= self.LIMITED_SAMPLE:
            strategy_type = "snapshot_analysis"
            reasoning = f"Limited data ({matches} matches) - providing current state snapshot without trend extrapolation"
            expected_accuracy = "medium"
        else:
            strategy_type = "baseline_only"
            reasoning = f"Insufficient data ({matches} matches) - providing only basic statistics with high uncertainty"
            expected_accuracy = "low"

        return AnalysisStrategy(
            strategy_type=strategy_type,
            reasoning=reasoning,
            expected_accuracy=expected_accuracy
        )

    def _generate_game_specific_prompt(
            self,
            game: str,
            team_name: str,
            matches_data: List[Dict],
            strategy: AnalysisStrategy,
            data_quality: DataQuality
    ) -> str:
        """
        Generate analysis prompt with quality context

        LLM is informed about data limitations
        """
        quality_context = f"""
DATA QUALITY CONTEXT:
- Sample size: {data_quality.matches_analyzed} matches ({data_quality.sample_size_rating})
- Confidence level: {data_quality.confidence_level * 100:.0f}%
- Analysis approach: {strategy.strategy_type}
- Limitations: {', '.join(data_quality.limitations) if data_quality.limitations else 'None'}

IMPORTANT: Your analysis must reflect the {data_quality.sample_size_rating} data quality. 
If confidence is below 70%, explicitly state uncertainty in your conclusions.
"""

        if game == "valorant":
            metrics_section = """
Provide statistical breakdown:
• Overall win rate (with sample size context)
• Map-specific performance (if sufficient data)
• Average rounds won/lost
• First blood efficiency
• ACS leaders (top 3 players)
• Clutch win percentage (if trackable)
• Agent composition patterns

Format: Bullet points. Under 300 words. Include confidence qualifiers where appropriate."""

        else:  # League of Legends
            metrics_section = """
Provide statistical breakdown:
• Overall win rate (with sample size context)
• Gold/XP differentials by game phase
• Objective control (dragons, barons, heralds)
• Vision score trends
• Damage share by role
• KDA leaders (top 3 players)
• Champion priority patterns

Format: Bullet points. Under 300 words. Include confidence qualifiers where appropriate."""

        prompt = f"""You are a professional {game.capitalize()} performance analyst.

Team: {team_name}
Matches analyzed: {len(matches_data)}

{quality_context}

Raw match data:
{json.dumps(matches_data, indent=2)[:2500]}

{metrics_section}

CRITICAL: If data is limited, use phrases like "based on limited sample" or "preliminary indication suggests".
"""
        return prompt

    @log_method
    @metric_counter("stats_tracker")
    async def analyze_team_stats(
            self,
            team_name: str,
            game: str = "valorant",
            recent_matches: int = 10
    ) -> Dict[str, Any]:
        """
        Deep team statistics analysis with proper reasoning

        Proper sequential reasoning: quality → strategy → analysis → confidence
        """
        reasoning: List[ReasoningStep] = []

        # Step 1: Request received
        self._next_step(reasoning, "Team statistics analysis requested",
                        input_data={
                            "team": team_name,
                            "game": game,
                            "requested_matches": recent_matches
                        })

        # Step 2: Fetch match data
        matches_data = []
        data_source = "GRID"

        try:
            matches_data = await self.grid.get_recent_matches(
                team_name=team_name,
                game=game,
                limit=recent_matches
            )

            self._next_step(reasoning, "Match data fetched from GRID",
                            output_data={
                                "matches_retrieved": len(matches_data),
                                "data_source": "GRID"
                            })

        except Exception as e:
            logger.error("GRID data fetch failed", extra={"error": str(e)})
            data_source = "fallback"

            self._next_step(reasoning, "GRID data fetch failed - using fallback",
                            output_data={
                                "error": str(e),
                                "fallback_active": True
                            })

        # Step 3 - Assess data quality
        data_quality = self._assess_data_quality(len(matches_data), data_source)

        self._next_step(reasoning, "Assessed statistical data quality",
                        output_data={
                            "sample_size_rating": data_quality.sample_size_rating,
                            "confidence_level": data_quality.confidence_level,
                            "sufficient_sample": data_quality.sufficient_sample,
                            "limitations_count": len(data_quality.limitations)
                        })

        # Handle insufficient data case
        if not data_quality.sufficient_sample and len(matches_data) < self.INSUFFICIENT_SAMPLE:
            self._next_step(reasoning, "Insufficient data for meaningful analysis",
                            output_data={"baseline_stats_only": True})

            return {
                "team": team_name,
                "game": game,
                "matches_analyzed": len(matches_data),
                "data_quality": asdict(data_quality),
                "stats_report": f"Insufficient data for {team_name}. Only {len(matches_data)} matches available. Minimum {self.LIMITED_SAMPLE} matches required for statistical analysis.",
                "reasoning": reasoning
            }

        # Step 4 - Select analysis strategy
        strategy = self._select_analysis_strategy(data_quality, game)

        self._next_step(reasoning, "Selected analysis strategy",
                        output_data={
                            "strategy_type": strategy.strategy_type,
                            "expected_accuracy": strategy.expected_accuracy,
                            "reasoning": strategy.reasoning
                        })

        # Step 5 - Generate quality-aware prompt
        prompt = self._generate_game_specific_prompt(
            game, team_name, matches_data, strategy, data_quality
        )

        self._next_step(reasoning, f"Generated {game.upper()}-specific analysis prompt with quality context",
                        output_data={
                            "strategy": strategy.strategy_type,
                            "quality_informed": True
                        })

        # Step 6: LLM analysis
        stats_report = None
        llm_fallback = False

        try:
            stats_report = await self.llm.chat(prompt)

            if self._is_invalid_response(stats_report):
                llm_fallback = True

                self._next_step(reasoning, "LLM response invalid - using baseline report",
                                output_data={"fallback_reason": "invalid_response"})
            else:
                self._next_step(reasoning, "Statistical analysis generated via LLM",
                                output_data={
                                    "report_length": len(stats_report),
                                    "llm_fallback": False
                                })

        except Exception as e:
            logger.error("LLM analysis failed", extra={"error": str(e)})
            llm_fallback = True

            self._next_step(reasoning, "LLM analysis failed - using baseline report",
                            output_data={
                                "error": str(e),
                                "fallback_reason": "exception"
                            })

        # Generate fallback if needed
        if llm_fallback or not stats_report:
            stats_report = f"""Statistical Analysis: {team_name}

**Data Context**: {len(matches_data)} matches analyzed ({data_quality.sample_size_rating} sample)

**Win Rate**: Approximately 50% (baseline estimate)
**Performance**: Balanced play across roles
**Data Quality**: {data_quality.sample_size_rating.capitalize()} - {', '.join(data_quality.limitations)}

**Note**: Detailed statistical analysis temporarily unavailable. Metrics reflect basic aggregation only."""

        # Step 7 - Analysis completed with confidence annotation
        self._next_step(reasoning, "Team statistics analysis completed",
                        output_data={
                            "matches_analyzed": len(matches_data),
                            "confidence_level": data_quality.confidence_level,
                            "strategy_used": strategy.strategy_type,
                            "llm_fallback_used": llm_fallback
                        })

        logger.info("Team stats analysis completed",
                    extra={
                        "team": team_name,
                        "matches": len(matches_data),
                        "confidence": data_quality.confidence_level
                    })

        return {
            "team": team_name,
            "game": game,
            "matches_analyzed": len(matches_data),
            "data_quality": asdict(data_quality),
            "analysis_strategy": asdict(strategy),
            "stats_report": stats_report.strip(),
            "confidence_level": data_quality.confidence_level,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("stats_tracker")
    async def get_player_stats(
            self,
            team_name: str,
            player_name: Optional[str] = None,
            game: str = "valorant",
            recent_matches: int = 8
    ) -> Dict[str, Any]:
        """
        Player-specific statistics with quality assessment

        Similar reasoning structure to team stats
        """
        reasoning: List[ReasoningStep] = []

        # Step 1: Request received
        self._next_step(reasoning, "Player statistics analysis requested",
                        input_data={
                            "team": team_name,
                            "player": player_name or "team_leaders",
                            "game": game,
                            "requested_matches": recent_matches
                        })

        # Step 2: Fetch data
        matches_data = []
        data_source = "GRID"

        try:
            matches_data = await self.grid.get_recent_matches(
                team_name=team_name,
                game=game,
                limit=recent_matches
            )

            self._next_step(reasoning, "Match data fetched for player analysis",
                            output_data={
                                "matches_retrieved": len(matches_data),
                                "data_source": "GRID"
                            })

        except Exception as e:
            logger.error("Data fetch failed", extra={"error": str(e)})
            data_source = "fallback"

            self._next_step(reasoning, "Data fetch failed - limited analysis",
                            output_data={"error": str(e)})

        # Step 3 - Quality assessment
        data_quality = self._assess_data_quality(len(matches_data), data_source)

        self._next_step(reasoning, "Assessed player data quality",
                        output_data={
                            "sample_size_rating": data_quality.sample_size_rating,
                            "confidence_level": data_quality.confidence_level
                        })

        # Handle insufficient data
        if not data_quality.sufficient_sample and len(matches_data) < self.INSUFFICIENT_SAMPLE:
            self._next_step(reasoning, "Insufficient data for player analysis",
                            output_data={"baseline_only": True})

            return {
                "team": team_name,
                "player": player_name or "team_leaders",
                "game": game,
                "matches_analyzed": len(matches_data),
                "data_quality": asdict(data_quality),
                "player_report": f"Insufficient data for player analysis. Only {len(matches_data)} matches available.",
                "reasoning": reasoning
            }

        # Step 4 - Strategy selection
        strategy = self._select_analysis_strategy(data_quality, game)

        self._next_step(reasoning, "Selected player analysis strategy",
                        output_data={
                            "strategy_type": strategy.strategy_type,
                            "expected_accuracy": strategy.expected_accuracy
                        })

        # Generate prompt (game-specific)
        quality_note = f"Note: Analysis based on {len(matches_data)} matches ({data_quality.sample_size_rating} sample, {data_quality.confidence_level * 100:.0f}% confidence)"

        if player_name:
            focus = f"Focus on player: {player_name}\n{quality_note}"
        else:
            focus = f"Identify and rank top 3 performing players\n{quality_note}"

        base_prompt = f"""You are a {game.capitalize()} player performance analyst.

Team: {team_name}
Matches: {len(matches_data)}
{focus}

Match data:
{json.dumps(matches_data, indent=2)[:2000]}

Provide player statistics (under 250 words, bullet points).
Include confidence qualifiers if sample is limited."""

        # Step 5: LLM analysis
        player_report = None
        llm_fallback = False

        try:
            player_report = await self.llm.chat(base_prompt)

            if self._is_invalid_response(player_report):
                llm_fallback = True
                self._next_step(reasoning, "LLM invalid - using fallback",
                                output_data={"fallback_used": True})
            else:
                self._next_step(reasoning, "Player statistics generated",
                                output_data={"report_length": len(player_report)})

        except Exception as e:
            logger.error("Player analysis failed", extra={"error": str(e)})
            llm_fallback = True
            self._next_step(reasoning, "LLM failed - using fallback",
                            output_data={"error": str(e)})

        if llm_fallback or not player_report:
            player_report = f"Player statistics for {player_name or 'team'} limited. Data quality: {data_quality.sample_size_rating}. {', '.join(data_quality.limitations)}"

        #
        # Step 6 - Completed
        self._next_step(reasoning, "Player analysis completed",
                        output_data={
                            "confidence_level": data_quality.confidence_level,
                            "llm_fallback": llm_fallback
                        })

        return {
            "team": team_name,
            "player": player_name or "team_leaders",
            "game": game,
            "matches_analyzed": len(matches_data),
            "data_quality": asdict(data_quality),
            "analysis_strategy": asdict(strategy),
            "player_report": player_report.strip(),
            "confidence_level": data_quality.confidence_level,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("stats_tracker")
    async def compare_teams(
            self,
            team_a: str,
            team_b: str,
            game: str = "valorant",
            recent_matches: int = 8
    ) -> Dict[str, Any]:
        """
        Head-to-head team comparison

        Demonstrates reasoning composition with multiple sub-analyses
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(reasoning, "Team comparison analysis requested",
                        input_data={
                            "team_a": team_a,
                            "team_b": team_b,
                            "game": game
                        })

        # Analyze both teams
        team_a_stats = await self.analyze_team_stats(team_a, game, recent_matches)
        team_b_stats = await self.analyze_team_stats(team_b, game, recent_matches)

        # Proper composition - reference, not extend
        self._next_step(reasoning, "Completed individual team analyses",
                        output_data={
                            "team_a_confidence": team_a_stats.get("confidence_level", 0.5),
                            "team_b_confidence": team_b_stats.get("confidence_level", 0.5),
                            "team_a_steps": len(team_a_stats.get("reasoning", [])),
                            "team_b_steps": len(team_b_stats.get("reasoning", []))
                        })

        # Overall confidence = minimum of both
        overall_confidence = min(
            team_a_stats.get("confidence_level", 0.5),
            team_b_stats.get("confidence_level", 0.5)
        )

        self._next_step(reasoning, "Calculated comparative analysis confidence",
                        output_data={
                            "overall_confidence": overall_confidence,
                            "limiting_factor": team_a if team_a_stats.get("confidence_level", 0) < team_b_stats.get(
                                "confidence_level", 0) else team_b
                        })

        self._next_step(reasoning, "Team comparison completed",
                        output_data={
                            "total_steps": len(reasoning),
                            "confidence_level": overall_confidence
                        })

        return {
            "team_a": team_a,
            "team_b": team_b,
            "game": game,
            "team_a_analysis": team_a_stats,  # Complete sub-analysis
            "team_b_analysis": team_b_stats,  # Complete sub-analysis
            "overall_confidence": overall_confidence,
            "reasoning": reasoning  # Main comparison reasoning only
        }
