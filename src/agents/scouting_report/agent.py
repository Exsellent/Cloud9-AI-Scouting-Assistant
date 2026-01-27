import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from shared.grid_client import GRIDClient
from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("scouting_report_agent")


@dataclass
class ReportValidation:
    """Validation results for scouting report quality"""
    word_count: int
    under_limit: bool
    has_strengths: bool
    has_weaknesses: bool
    has_recommendations: bool
    confidence: float
    data_reliability: str  # "high", "medium", "low"


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


class ScoutingReportAgent(MCPAgent):
    """
    Professional Esports Scouting Report Agent

    Key Features:
    1. Data-Driven Analysis: Real match data from GRID API
    2. Game-Aware Reasoning: Adapts prompts for Valorant/LoL
    3. Deterministic Validation: Word count, structure, confidence scoring
    4. Insight Extraction: Structured key findings for coaches
    5. Reliability Tracking: Data quality assessment

    Proper sequential reasoning, validation steps, confidence scoring

    This is a DECISION-SUPPORT agent, not just a text generator.
    """

    # Validation thresholds
    MAX_WORD_COUNT = 300
    MIN_WORD_COUNT = 100
    MIN_MATCHES_FOR_HIGH_CONFIDENCE = 5

    def __init__(self):
        super().__init__("ScoutingReport")
        self.llm = LLMClient()
        self.grid = GRIDClient()

        self.register_tool("generate_scout_report", self.generate_scout_report)
        self.register_tool("validate_report", self.validate_report)
        self.register_tool("extract_tactical_insights", self.extract_tactical_insights)

        logger.info("ScoutingReportAgent initialized with GRID integration")

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
            "for more information check", "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    def _validate_report_quality(
            self,
            report: str,
            matches_analyzed: int
    ) -> ReportValidation:
        """
         Deterministic validation of scouting report quality

        Checks:
        - Word count within limits
        - Required sections present
        - Data reliability based on sample size
        """
        words = report.split()
        word_count = len(words)

        # Check for required sections
        report_lower = report.lower()
        has_strengths = any(keyword in report_lower for keyword in [
            "strength", "excel", "strong", "advantage", "proficient"
        ])
        has_weaknesses = any(keyword in report_lower for keyword in [
            "weakness", "vulnerability", "exploit", "struggle", "inconsistent"
        ])
        has_recommendations = any(keyword in report_lower for keyword in [
            "recommend", "counter", "strategy", "ban", "pick", "focus"
        ])

        # Calculate data reliability
        if matches_analyzed >= self.MIN_MATCHES_FOR_HIGH_CONFIDENCE:
            data_reliability = "high"
            base_confidence = 0.8
        elif matches_analyzed >= 3:
            data_reliability = "medium"
            base_confidence = 0.6
        else:
            data_reliability = "low"
            base_confidence = 0.4

        # Adjust confidence based on content quality
        confidence = base_confidence
        if self.MIN_WORD_COUNT <= word_count <= self.MAX_WORD_COUNT:
            confidence += 0.1
        if has_strengths and has_weaknesses:
            confidence += 0.05
        if has_recommendations:
            confidence += 0.05

        return ReportValidation(
            word_count=word_count,
            under_limit=word_count <= self.MAX_WORD_COUNT,
            has_strengths=has_strengths,
            has_weaknesses=has_weaknesses,
            has_recommendations=has_recommendations,
            confidence=min(confidence, 0.95),
            data_reliability=data_reliability
        )

    def _extract_insights(self, report: str) -> List[str]:
        """
        Extract key tactical insights from report

        Better parsing with context awareness
        """
        lines = report.split('\n')
        insights = []

        # Extract bullet points and numbered lists
        for line in lines:
            line = line.strip()

            if any(line.startswith(prefix) for prefix in ['•', '-', '*', '1.', '2.', '3.', '4.', '5.']):
                # Remove the prefix
                cleaned = line.lstrip('•-*0123456789. ')
                if cleaned and len(cleaned) > 20:
                    insights.append(cleaned)

        # If no structured list found, try to extract sentences with keywords
        if not insights:
            for line in lines:
                if any(keyword in line.lower() for keyword in [
                    'recommend', 'should', 'focus', 'exploit', 'ban', 'pick', 'counter'
                ]):
                    insights.append(line.strip())

        return insights[:8]  # Top 8 insights

    def _calculate_confidence(
            self,
            matches_analyzed: int,
            llm_fallback: bool,
            validation: ReportValidation
    ) -> float:
        """
        Calculate overall confidence score

        Combines:
        - Data availability (matches)
        - LLM reliability
        - Validation quality
        """
        if llm_fallback:
            return 0.3  # Low confidence for fallback

        # Base confidence from validation
        confidence = validation.confidence

        # Penalty for insufficient data
        if matches_analyzed < 3:
            confidence *= 0.7

        return min(confidence, 0.95)

    @log_method
    @metric_counter("scouting_report")
    async def generate_scout_report(
            self,
            opponent_team: str,
            game: str = "valorant",
            recent_matches: int = 5
    ) -> Dict[str, Any]:
        """
        Generate automated scouting report for opponent team

        Proper sequential reasoning with validation and confidence scoring
        """
        reasoning: List[ReasoningStep] = []

        # Step 1: Request received
        self._next_step(reasoning, "Scouting report generation requested",
                        input_data={
                            "opponent": opponent_team,
                            "game": game,
                            "requested_matches": recent_matches
                        })

        # Step 2: Fetch match data from GRID
        matches_data = []
        grid_error = None

        try:
            matches_data = await self.grid.get_recent_matches(
                team_name=opponent_team,
                game=game,
                limit=recent_matches
            )

            self._next_step(reasoning, "Match data fetched from GRID successfully",
                            output_data={
                                "matches_found": len(matches_data),
                                "data_source": "GRID",
                                "data_quality": "high" if len(matches_data) >= 5 else "medium"
                            })

        except Exception as e:
            logger.error("GRID data fetch failed", extra={"error": str(e)})
            grid_error = str(e)

            #  Explicit GRID error step
            self._next_step(reasoning, "GRID data fetch failed - proceeding with limited data",
                            output_data={
                                "error": str(e),
                                "fallback_data": True
                            })

        # Step 3: Generate game-specific prompt
        if game.lower() == "valorant":
            prompt = f"""You are a professional Valorant esports analyst preparing a concise scouting report.

Opponent: {opponent_team}
Matches analyzed: {len(matches_data)}

Raw match data:
{json.dumps(matches_data, indent=2)[:3000]}

Generate a coach-friendly report (under 300 words) covering:
• Preferred agent compositions (most frequent 3-5 agents)
• Map-specific defaults and site setups
• Key players and their roles/tendencies
• Strengths (what they excel at)
• Weaknesses (exploitable patterns)
• Counter-strategy recommendations (agents, executes, map bans)

Tone: Professional, data-driven, actionable.
Format: Use bullet points for key insights."""

        else:  # League of Legends
            prompt = f"""You are a professional League of Legends esports analyst preparing a concise scouting report.

Opponent: {opponent_team}
Matches analyzed: {len(matches_data)}

Raw match data:
{json.dumps(matches_data, indent=2)[:3000]}

Generate a coach-friendly report (under 300 words) covering:
• Champion pools per role (top 3-4 per role)
• Common draft patterns and priority picks/bans
• Early/mid/late game tendencies
• Win conditions and team fight preferences
• Weaknesses (exploitable patterns)
• Counter-strategy recommendations (draft, macro play)

Tone: Professional, data-driven, actionable.
Format: Use bullet points for key insights."""

        self._next_step(reasoning, f"Generated {game.upper()}-specific analysis prompt",
                        output_data={
                            "game": game,
                            "prompt_type": "game_aware",
                            "data_points": len(matches_data)
                        })

        # Step 4: LLM analysis
        report = None
        llm_fallback = False

        try:
            report = await self.llm.chat(prompt)

            #  Check for LLM errors
            if self._is_invalid_response(report):
                llm_fallback = True

                # Explicit LLM error fallback step
                self._next_step(reasoning, "LLM response invalid - using baseline report",
                                output_data={
                                    "fallback_reason": "invalid_response",
                                    "llm_error": True
                                })
            else:
                # Successful LLM response
                self._next_step(reasoning, "LLM analysis completed successfully",
                                output_data={
                                    "initial_length": len(report),
                                    "initial_word_count": len(report.split())
                                })

        except Exception as e:
            logger.error("LLM analysis failed", extra={"error": str(e)})
            llm_fallback = True

            #  Explicit exception fallback step
            self._next_step(reasoning, "LLM request failed - using baseline report",
                            output_data={
                                "error": str(e),
                                "fallback_reason": "exception"
                            })

        # Generate fallback if needed
        if llm_fallback or not report:
            report = f"""Scouting Report: {opponent_team}

**Data Availability**: Limited match data available ({len(matches_data)} matches analyzed).

**General Assessment**:
• Team demonstrates balanced playstyle across roles
• Consistent execution in standard compositions
• No clear exploitable patterns identified with current data sample

**Recommendations**:
• Request additional match data for comprehensive analysis
• Focus on standard counter-strategies
• Monitor live performance for adaptation opportunities

**Note**: This is a baseline assessment. For detailed tactical insights, additional match data is required."""

        # Step 5 - Validate report quality
        validation = self._validate_report_quality(report, len(matches_data))

        self._next_step(reasoning, "Scouting report validated against quality criteria",
                        output_data={
                            "word_count": validation.word_count,
                            "under_limit": validation.under_limit,
                            "has_strengths": validation.has_strengths,
                            "has_weaknesses": validation.has_weaknesses,
                            "data_reliability": validation.data_reliability
                        })

        # Step 6 - Extract tactical insights
        key_insights = self._extract_insights(report)

        self._next_step(reasoning, "Extracted key tactical insights from report",
                        output_data={
                            "insights_extracted": len(key_insights),
                            "structured_data_available": True
                        })

        # Step 7 - Calculate overall confidence
        overall_confidence = self._calculate_confidence(
            matches_analyzed=len(matches_data),
            llm_fallback=llm_fallback,
            validation=validation
        )

        # Step 8 - Report finalized (ALWAYS present)
        self._next_step(reasoning, "Scouting report generation completed",
                        output_data={
                            "final_word_count": validation.word_count,
                            "validation_passed": validation.under_limit and validation.has_recommendations,
                            "confidence_level": overall_confidence,
                            "data_reliability": validation.data_reliability,
                            "llm_fallback_used": llm_fallback,
                            "grid_error": bool(grid_error)
                        })

        logger.info("Scouting report completed",
                    extra={
                        "opponent": opponent_team,
                        "game": game,
                        "matches": len(matches_data),
                        "confidence": overall_confidence
                    })

        return {
            "opponent": opponent_team,
            "game": game,
            "matches_analyzed": len(matches_data),
            "report": report.strip(),
            "key_insights": key_insights,
            "validation": asdict(validation),
            "confidence_level": overall_confidence,
            "data_source": "GRID" if matches_data else "fallback",
            "llm_fallback_used": llm_fallback,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("scouting_report")
    async def validate_report(self, report: str, matches_analyzed: int) -> Dict[str, Any]:
        """
        Standalone report validation tool

        Allows external validation of scouting report quality
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(reasoning, "Report validation requested",
                        input_data={
                            "report_length": len(report),
                            "matches_count": matches_analyzed
                        })

        validation = self._validate_report_quality(report, matches_analyzed)

        self._next_step(reasoning, "Validation completed",
                        output_data=asdict(validation))

        return {
            "validation": asdict(validation),
            "passed": validation.under_limit and validation.has_recommendations,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("scouting_report")
    async def extract_tactical_insights(
            self,
            report: str,
            game: str = "valorant"
    ) -> Dict[str, Any]:
        """
        Extract structured tactical insights from report

        Game-aware extraction for better downstream usage
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(reasoning, "Tactical insights extraction requested",
                        input_data={
                            "report_length": len(report),
                            "game": game
                        })

        insights = self._extract_insights(report)

        # Categorize insights by type
        categorized = {
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "general": []
        }

        for insight in insights:
            insight_lower = insight.lower()
            if any(keyword in insight_lower for keyword in ["strength", "excel", "strong"]):
                categorized["strengths"].append(insight)
            elif any(keyword in insight_lower for keyword in ["weakness", "exploit", "struggle"]):
                categorized["weaknesses"].append(insight)
            elif any(keyword in insight_lower for keyword in ["recommend", "counter", "ban", "pick"]):
                categorized["recommendations"].append(insight)
            else:
                categorized["general"].append(insight)

        self._next_step(reasoning, "Insights categorized by tactical type",
                        output_data={
                            "strengths_found": len(categorized["strengths"]),
                            "weaknesses_found": len(categorized["weaknesses"]),
                            "recommendations_found": len(categorized["recommendations"])
                        })

        self._next_step(reasoning, "Tactical insights extraction completed",
                        output_data={"total_insights": len(insights)})

        return {
            "insights": insights,
            "categorized": categorized,
            "game": game,
            "reasoning": reasoning
        }
