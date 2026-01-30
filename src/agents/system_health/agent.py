import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional

import httpx

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("system_health_agent")


@dataclass
class AgentHealthStatus:
    """Individual agent health assessment with enhanced metrics"""
    agent_name: str
    status: str  # "OK", "WARNING", "CRITICAL", "DOWN"
    health_score: int  # 0-100
    error_rate: float
    tasks_processed: int
    errors: int
    response_time_ms: Optional[float]
    reachable: bool
    trend: str  # "stable", "deteriorating", "recovering"
    trend_confidence: float  # 0.0-1.0
    issues: List[str]
    recommendations: List[str]


@dataclass
class SystemHealthStatus:
    """Overall system health assessment"""
    overall_status: str  # "OK", "DEGRADED", "CRITICAL", "DOWN"
    health_score_avg: float  # Metric only, not status determinant
    total_agents: int
    ok_agents: int
    warning_agents: int
    critical_agents: int
    down_agents: int
    systemic_risk: bool
    cascade_detected: bool
    cascades: List[str]
    prioritized_actions: List[Dict[str, Any]]
    issues: List[str]
    timestamp: str


def log_method(func):
    """Decorator for logging method calls with timing"""

    async def wrapper(self, *args, **kwargs):
        start = datetime.now()
        logger.info(f"{func.__name__} called")
        try:
            result = await func(self, *args, **kwargs)
            duration = (datetime.now() - start).total_seconds()
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {str(e)}", exc_info=True)
            raise

    return wrapper


class SystemHealthAgent(MCPAgent):
    """
    Production-Ready System Health Monitoring Agent

    This agent implements intelligent DevOps monitoring with:
    - Deterministic decision-making (not LLM-based)
    - Metrics-driven health scoring
    - Temporal analysis (trends)
    - Dependency-aware cascade detection
    - Priority-based system status (worst-case principle)
    - AI as explainability layer only
    """

    # Health scoring thresholds
    ERROR_RATE_CRITICAL = 0.15  # 15% error rate
    ERROR_RATE_WARNING = 0.05  # 5% error rate
    LATENCY_CRITICAL_MS = 1500  # 1.5 seconds
    LATENCY_WARNING_MS = 800  # 800ms
    MIN_ACTIVITY_THRESHOLD = 5  # Minimum tasks for reliable stats

    # Trend detection
    TREND_DETERIORATING_SLOPE = -2.0  # Score declining at 2 points per check
    TREND_RECOVERING_SLOPE = 2.0  # Score improving at 2 points per check
    MIN_HISTORY_FOR_TREND = 3  # Minimum data points for trend
    HISTORY_WINDOW_SIZE = 24  # Keep last 24 checks (~24 hours if hourly)

    # System-level thresholds
    SYSTEMIC_RISK_THRESHOLD = 2  # 2+ critical/down agents = systemic
    HTTP_TIMEOUT = 8.0

    # Docker service URLs (Cloud9 esports platform)
    KNOWN_AGENTS = {
        "ScoutingReport": "http://scouting_report:8404/health",
        "DraftCoach": "http://draft_coach:8401/health",
        "CounterPlay": "http://counter_play:8403/health",
        "MatchHistory": "http://match_history:8402/health",
        "StatsTracker": "http://stats_tracker:8407/health",
    }

    # Dependency graph for cascade detection
    # Format: {dependent: [dependencies]}
    DEPENDENCY_GRAPH = {
        "ScoutingReport": ["MatchHistory", "StatsTracker"],
        "DraftCoach": ["ScoutingReport", "MatchHistory"],
        "CounterPlay": ["ScoutingReport", "DraftCoach"],
        "MatchHistory": [],  # No dependencies (root)
        "StatsTracker": [],  # No dependencies (root)
    }

    def __init__(self, metrics_agent_url: Optional[str] = None,
                 redis_client=None):
        """
        Initialize SystemHealthAgent

        Args:
            metrics_agent_url: URL for MetricsAgent integration (optional)
            redis_client: Redis client for persistent history (optional)
        """
        super().__init__("SystemHealth")
        self.llm = LLMClient()

        # Metrics integration
        self.metrics_agent_url = metrics_agent_url or "http://metrics_agent:8307"

        # Dynamic history tracking (in-memory with optional Redis backup)
        self.agent_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.redis = redis_client  # Optional persistent storage

        # Register tools
        self.register_tool("full_system_check", self.full_system_check)
        self.register_tool("quick_ping", self.quick_ping)
        self.register_tool("diagnose_agent", self.diagnose_agent)
        self.register_tool("get_system_status", self.get_system_status)
        self.register_tool("get_agent_history", self.get_agent_history)
        self.register_tool("analyze_anomalies", self.analyze_anomalies)

        logger.info("SystemHealthAgent initialized",
                    extra={
                        "agents_monitored": len(self.KNOWN_AGENTS),
                        "metrics_integration": bool(metrics_agent_url),
                        "persistent_storage": bool(redis_client)
                    })

    def _next_step(self, reasoning: List[ReasoningStep], description: str,
                   input_data: Optional[Dict] = None,
                   output_data: Optional[Dict] = None):
        """Helper to add sequential reasoning steps"""
        reasoning.append(ReasoningStep(
            step_number=len(reasoning) + 1,
            description=description,
            input_data=input_data or {},
            output_data=output_data or {}
        ))

    async def _ping_agent(self, name: str, url: str) -> Dict[str, Any]:
        """
        HTTP health check for individual agent

        Returns:
            Dict with reachability, status code, and response time
        """
        try:
            async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT) as client:
                resp = await client.get(url)
                return {
                    "reachable": True,
                    "status_code": resp.status_code,
                    "response_time_ms": round(resp.elapsed.total_seconds() * 1000, 2),
                    "healthy": resp.status_code == 200,
                    "details": resp.json() if resp.headers.get("content-type", "").startswith(
                        "application/json") else None
                }
        except httpx.TimeoutException:
            logger.warning(f"Agent {name} timeout")
            return {"reachable": False, "error": "timeout"}
        except httpx.RequestError as e:
            logger.warning(f"Agent {name} request error: {str(e)}")
            return {"reachable": False, "error": f"request_error: {str(e)}"}
        except Exception as e:
            logger.error(f"Agent {name} unexpected error: {str(e)}")
            return {"reachable": False, "error": f"unknown: {str(e)}"}

    async def _get_metrics_from_agent(self) -> Dict[str, Dict[str, int]]:
        """
        Get real metrics from MetricsAgent

        Real integration with fallback to mock

        Returns:
            Dict mapping agent_name to {tasks_processed, errors, latency_ms}
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self.metrics_agent_url}/get_all_metrics",
                    json={}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    metrics = data.get("metrics", {})
                    logger.info("Successfully retrieved metrics from MetricsAgent",
                                extra={"agents_count": len(metrics)})
                    return metrics
                else:
                    logger.warning(f"MetricsAgent returned {resp.status_code}")
                    return self._get_mock_metrics()
        except Exception as e:
            logger.warning(f"MetricsAgent unavailable: {str(e)} - using mock data")
            return self._get_mock_metrics()

    def _get_mock_metrics(self) -> Dict[str, Dict[str, int]]:
        """
        Fallback mock metrics for development/testing

        In production, this should only be used when MetricsAgent is unavailable
        """
        return {
            "ScoutingReport": {"tasks_processed": 145, "errors": 7, "latency_ms": 620},
            "DraftCoach": {"tasks_processed": 98, "errors": 3, "latency_ms": 450},
            "CounterPlay": {"tasks_processed": 67, "errors": 12, "latency_ms": 1350},
            "MatchHistory": {"tasks_processed": 203, "errors": 2, "latency_ms": 280},
            "StatsTracker": {"tasks_processed": 178, "errors": 1, "latency_ms": 310},
        }

    def _compute_health_score(
            self,
            error_rate: float,
            latency_ms: Optional[float],
            tasks_processed: int,
            reachable: bool
    ) -> int:
        """
        Compute multi-factor health score (0-100)

        #3: Unreachable agents return 0 immediately (hard gate)

        Factors:
        - Reachability (critical): if False → instant 0
        - Error rate: -40 (critical) or -20 (warning)
        - Latency: -30 (critical) or -15 (warning)
        - Activity: -10 if insufficient data

        Returns:
            Integer score from 0 to 100
        """
        # 3: If unreachable, other metrics are irrelevant
        if not reachable:
            return 0  # Instant DOWN status, no further calculation

        score = 100

        # Error rate penalty
        if error_rate >= self.ERROR_RATE_CRITICAL:
            score -= 40
        elif error_rate >= self.ERROR_RATE_WARNING:
            score -= 20

        # Latency penalty
        if latency_ms is not None:
            if latency_ms >= self.LATENCY_CRITICAL_MS:
                score -= 30
            elif latency_ms >= self.LATENCY_WARNING_MS:
                score -= 15

        # Low activity penalty (less reliable stats)
        # Note: This is a conservative approach - prefer caution over false confidence
        if tasks_processed < self.MIN_ACTIVITY_THRESHOLD:
            score -= 10

        return max(0, min(100, score))

    def _classify_status(self, score: int) -> str:
        """
        Classify health status based on score

        Note: This is for INDIVIDUAL agents. System status uses different logic.

        Returns:
            "OK", "WARNING", "CRITICAL", or "DOWN"
        """
        if score >= 80:
            return "OK"
        elif score >= 50:
            return "WARNING"
        elif score >= 20:
            return "CRITICAL"
        else:
            return "DOWN"

    def _detect_trend(
            self,
            agent_name: str,
            current_score: int
    ) -> tuple[str, float]:
        """
        Detect health trend using linear regression

        #1: Uses all history points, not just last point

        Algorithm:
        - Collect historical scores + current
        - Calculate linear regression slope
        - Classify: deteriorating (slope < -2), recovering (slope > 2), stable

        Returns:
            Tuple of (trend_label, confidence)
            trend_label: "deteriorating", "stable", "recovering"
            confidence: 0.0-1.0 based on data points available
        """
        history = self.agent_history.get(agent_name, [])

        # Need at least MIN_HISTORY_FOR_TREND points
        if len(history) < self.MIN_HISTORY_FOR_TREND:
            return "stable", 0.0

        # Extract scores from history
        scores = [h["score"] for h in history] + [current_score]
        n = len(scores)

        # Linear regression: y = mx + b, we only need slope (m)
        x = list(range(n))
        y = scores

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        # Calculate slope using least squares
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable", 0.0

        slope = numerator / denominator

        # Confidence based on amount of data (more data = higher confidence)
        confidence = min(1.0, len(history) / 10.0)  # Full confidence at 10+ points

        # Classify trend based on slope
        if slope <= self.TREND_DETERIORATING_SLOPE:
            return "deteriorating", confidence
        elif slope >= self.TREND_RECOVERING_SLOPE:
            return "recovering", confidence
        else:
            return "stable", confidence

    def _update_agent_history(
            self,
            agent_name: str,
            score: int,
            status: str,
            error_rate: float,
            latency_ms: Optional[float]
    ):
        """
        Update rolling window history for agent

        #1: History is now dynamic, not static

        Stores in memory and optionally backs up to Redis for persistence
        """
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "status": status,
            "error_rate": error_rate,
            "latency_ms": latency_ms
        }

        # Add to in-memory history
        self.agent_history[agent_name].append(history_entry)

        # Maintain rolling window (keep only last N points)
        if len(self.agent_history[agent_name]) > self.HISTORY_WINDOW_SIZE:
            self.agent_history[agent_name].pop(0)

        # Optional: Persist to Redis for survival across restarts
        if self.redis:
            try:
                key = f"health_history:{agent_name}"
                self.redis.lpush(key, str(history_entry))
                self.redis.ltrim(key, 0, self.HISTORY_WINDOW_SIZE - 1)
            except Exception as e:
                logger.warning(f"Failed to persist history to Redis: {e}")

    def _detect_cascades(
            self,
            agent_healths: Dict[str, AgentHealthStatus]
    ) -> List[str]:
        """
        Detect cascade failures based on dependency graph

        Logic: Only check for cascades when agent is already unhealthy

        Returns:
            List of cascade descriptions
        """
        cascades = []

        for agent, dependencies in self.DEPENDENCY_GRAPH.items():
            if agent not in agent_healths:
                continue

            agent_health = agent_healths[agent]

            # Only check if this agent is unhealthy
            if agent_health.status in ["CRITICAL", "DOWN"]:
                # Check if any dependencies are also unhealthy
                unhealthy_deps = []
                for dep in dependencies:
                    if dep in agent_healths:
                        dep_health = agent_healths[dep]
                        if dep_health.status in ["WARNING", "CRITICAL", "DOWN"]:
                            unhealthy_deps.append(f"{dep} ({dep_health.status})")

                if unhealthy_deps:
                    cascade_msg = (
                        f"{agent} ({agent_health.status}) degraded possibly due to: "
                        f"{', '.join(unhealthy_deps)}"
                    )
                    cascades.append(cascade_msg)

        return cascades

    def _determine_system_status(
            self,
            agent_healths: Dict[str, AgentHealthStatus],
            critical_down_count: int
    ) -> str:
        """
        Determine overall system status using priority-based logic

        #4: CRITICAL - System status based on worst-case, NOT average

        Priority order: DOWN > CRITICAL > WARNING > OK

        Logic:
        - Any agent DOWN → system DOWN
        - 2+ agents CRITICAL/DOWN → system CRITICAL (systemic risk)
        - 1 agent CRITICAL → system DEGRADED
        - Any agent WARNING → system DEGRADED
        - All OK → system OK

        This reflects production reality: one critical failure affects the whole system.

        Args:
            agent_healths: Dict of all agent health statuses
            critical_down_count: Number of agents in CRITICAL or DOWN state

        Returns:
            "OK", "DEGRADED", "CRITICAL", or "DOWN"
        """
        # Priority 1: Any agent DOWN means system is DOWN
        if any(h.status == "DOWN" for h in agent_healths.values()):
            return "DOWN"

        # Priority 2: Multiple critical = systemic risk
        if critical_down_count >= self.SYSTEMIC_RISK_THRESHOLD:
            return "CRITICAL"

        # Priority 3: Single critical = degraded but contained
        if critical_down_count >= 1:
            return "DEGRADED"

        # Priority 4: Any warning = degraded (needs monitoring)
        if any(h.status == "WARNING" for h in agent_healths.values()):
            return "DEGRADED"

        # All agents OK
        return "OK"

    def _generate_recommendations(
            self,
            health: AgentHealthStatus
    ) -> List[str]:
        """
        Generate agent-specific recommendations based on health status

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        if health.status == "DOWN":
            recommendations.extend([
                "IMMEDIATE: Restart Docker container",
                "Check container logs: docker logs <container>",
                "Verify network connectivity",
                "Check resource limits (CPU/Memory)"
            ])
        elif health.status == "CRITICAL":
            recommendations.extend([
                "HIGH PRIORITY: Investigate error logs",
                "Check database connections",
                "Verify external API dependencies",
                "Consider rolling back recent deployment"
            ])
        elif health.status == "WARNING":
            recommendations.extend([
                "MONITOR: Increase monitoring frequency",
                "Review recent configuration changes",
                "Check for gradual resource exhaustion",
                "Analyze error patterns in logs"
            ])
        else:  # OK
            recommendations.append("Operating normally - maintain current monitoring")

        # Trend-specific recommendations
        if health.trend == "deteriorating":
            recommendations.insert(0, f"⚠️ ALERT: Health deteriorating (confidence: {health.trend_confidence:.0%})")
        elif health.trend == "recovering":
            recommendations.append(f"✓ Recovering (confidence: {health.trend_confidence:.0%})")

        # Performance-specific
        if health.response_time_ms and health.response_time_ms > self.LATENCY_WARNING_MS:
            recommendations.append(
                f"Performance: High latency detected ({health.response_time_ms:.0f}ms)"
            )

        if health.error_rate > self.ERROR_RATE_WARNING:
            recommendations.append(
                f"Errors: Elevated error rate ({health.error_rate * 100:.1f}%)"
            )

        return recommendations

    def _prioritize_actions(
            self,
            agent_healths: Dict[str, AgentHealthStatus],
            systemic_risk: bool,
            cascades: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate prioritized action plan

        Returns:
            List of actions sorted by severity (worst first)
        """
        actions = []

        # System-wide issues get highest priority
        if systemic_risk or cascades:
            actions.append({
                "priority": 0,
                "type": "SYSTEM_WIDE",
                "severity": "CRITICAL",
                "agent": "SYSTEM",
                "actions": ["⚠️ SYSTEMIC ISSUE DETECTED"] + cascades,
                "immediate": True
            })

        # Sort agents by health score (worst first)
        sorted_agents = sorted(
            agent_healths.values(),
            key=lambda h: (h.status != "DOWN", h.status != "CRITICAL", h.health_score)
        )

        for health in sorted_agents:
            # Determine severity and priority
            if health.status == "DOWN":
                severity = "CRITICAL"
                priority = 1
            elif health.status == "CRITICAL":
                severity = "CRITICAL" if health.trend == "deteriorating" else "HIGH"
                priority = 2
            elif health.status == "WARNING":
                severity = "HIGH" if health.trend == "deteriorating" else "MEDIUM"
                priority = 3
            else:  # OK
                severity = "LOW"
                priority = 4

            action = {
                "priority": priority,
                "type": "AGENT",
                "severity": severity,
                "agent": health.agent_name,
                "status": health.status,
                "score": health.health_score,
                "trend": health.trend,
                "trend_confidence": health.trend_confidence,
                "actions": health.recommendations,
                "immediate": health.status in ["DOWN", "CRITICAL"]
            }
            actions.append(action)

        # Sort by priority (0 = highest)
        return sorted(actions, key=lambda a: a["priority"])

    @log_method
    @metric_counter("system_health")
    async def full_system_check(self) -> Dict[str, Any]:
        """
        Comprehensive system health check with all intelligence layers

        Sequential flow:
        1. Collect data (HTTP ping + metrics)
        2. Compute scores
        3. Detect trends
        4. Update history
        5. Determine system status (priority-based)
        6. Detect cascades
        7. Generate actions
        8. LLM explanation (optional)

        Returns:
            Complete health assessment with reasoning
        """
        reasoning: List[ReasoningStep] = []
        self._next_step(reasoning, "Full system health check initiated")

        # Step 1: HTTP reachability check
        ping_results = {}
        for name, url in self.KNOWN_AGENTS.items():
            ping_results[name] = await self._ping_agent(name, url)

        reachable_count = sum(1 for r in ping_results.values() if r.get("reachable"))
        self._next_step(
            reasoning,
            "HTTP reachability check completed",
            output_data={
                "total_agents": len(ping_results),
                "reachable": reachable_count,
                "unreachable": len(ping_results) - reachable_count
            }
        )

        # Step 2: Get metrics (real or fallback)
        metrics = await self._get_metrics_from_agent()
        self._next_step(
            reasoning,
            "Metrics collected from MetricsAgent",
            output_data={
                "metrics_source": "MetricsAgent" if metrics else "mock_fallback",
                "agents_with_metrics": len(metrics)
            }
        )

        # Step 3: Compute health scores and classify
        agent_healths = {}

        for agent_name in self.KNOWN_AGENTS.keys():
            ping = ping_results.get(agent_name, {"reachable": False})
            agent_metrics = metrics.get(agent_name, {
                "tasks_processed": 0,
                "errors": 0,
                "latency_ms": None
            })

            # Calculate error rate
            tasks = agent_metrics.get("tasks_processed", 0)
            errors = agent_metrics.get("errors", 0)
            error_rate = errors / max(tasks, 1)

            # Compute health score
            score = self._compute_health_score(
                error_rate=error_rate,
                latency_ms=agent_metrics.get("latency_ms"),
                tasks_processed=tasks,
                reachable=ping.get("reachable", False)
            )

            # Classify status
            status = self._classify_status(score)

            # Detect trend (BEFORE updating history)
            trend, confidence = self._detect_trend(agent_name, score)

            # Update history (Dynamic history)
            self._update_agent_history(
                agent_name=agent_name,
                score=score,
                status=status,
                error_rate=error_rate,
                latency_ms=agent_metrics.get("latency_ms")
            )

            # Build health status
            health = AgentHealthStatus(
                agent_name=agent_name,
                status=status,
                health_score=score,
                error_rate=error_rate,
                tasks_processed=tasks,
                errors=errors,
                response_time_ms=ping.get("response_time_ms") or agent_metrics.get("latency_ms"),
                reachable=ping.get("reachable", False),
                trend=trend,
                trend_confidence=confidence,
                issues=[],
                recommendations=[]
            )

            # Generate recommendations
            health.recommendations = self._generate_recommendations(health)

            agent_healths[agent_name] = health

        self._next_step(
            reasoning,
            "Health scores computed and trends detected",
            output_data={
                "agents_scored": len(agent_healths),
                "ok": sum(1 for h in agent_healths.values() if h.status == "OK"),
                "warning": sum(1 for h in agent_healths.values() if h.status == "WARNING"),
                "critical": sum(1 for h in agent_healths.values() if h.status == "CRITICAL"),
                "down": sum(1 for h in agent_healths.values() if h.status == "DOWN")
            }
        )

        # Step 4: System-level assessment with priority-based status
        scores = [h.health_score for h in agent_healths.values()]
        avg_score = sum(scores) / len(scores) if scores else 0

        critical_down_count = sum(
            1 for h in agent_healths.values()
            if h.status in ["CRITICAL", "DOWN"]
        )

        # 4: Priority-based system status (NOT average-based)
        overall_status = self._determine_system_status(agent_healths, critical_down_count)
        systemic_risk = critical_down_count >= self.SYSTEMIC_RISK_THRESHOLD

        # Detect cascades
        cascades = self._detect_cascades(agent_healths)

        self._next_step(
            reasoning,
            "System-level health assessment completed",
            output_data={
                "overall_status": overall_status,
                "avg_score": round(avg_score, 1),
                "determination_method": "priority_based_worst_case",
                "systemic_risk": systemic_risk,
                "cascades_detected": len(cascades)
            }
        )

        # Step 5: Generate prioritized actions
        actions = self._prioritize_actions(agent_healths, systemic_risk, cascades)

        self._next_step(
            reasoning,
            "Prioritized action plan generated",
            output_data={
                "total_actions": len(actions),
                "immediate_actions": sum(1 for a in actions if a.get("immediate"))
            }
        )

        # Step 6: LLM explanation (optional, only if issues detected)
        llm_summary = None
        if overall_status != "OK":
            # Build structured prompt for LLM
            top_actions = actions[:5]
            action_summary = "\n".join([
                f"• {a['agent']}: {a['severity']} - {a.get('status', 'N/A')} "
                f"(score: {a.get('score', 'N/A')}, trend: {a.get('trend', 'N/A')})"
                for a in top_actions
            ])

            prompt = f"""DevOps System Health Report - Cloud9 Esports Platform

Overall Status: {overall_status} (Average Score: {avg_score:.1f}/100)
Note: Status is priority-based (worst-case), not average-based
Systemic Risk: {'YES - MULTIPLE CRITICAL FAILURES' if systemic_risk else 'NO'}
Cascade Failures: {len(cascades)} detected

Top Priority Actions:
{action_summary}

{('Cascades Detected:\n' + chr(10).join(f'• {c}' for c in cascades)) if cascades else ''}

Provide a concise executive summary with:
1. Most critical issues requiring immediate attention
2. Recommended remediation order
3. Potential root causes based on cascade patterns

Keep response under 200 words, technical and actionable."""

            try:
                llm_summary = await self.llm.chat(prompt)
                self._next_step(
                    reasoning,
                    "LLM executive summary generated",
                    output_data={"summary_length": len(llm_summary)}
                )
            except Exception as e:
                logger.warning(f"LLM summary failed: {e}")
                llm_summary = "LLM explanation unavailable - proceed with action plan"
                self._next_step(
                    reasoning,
                    "LLM summary generation failed",
                    output_data={"error": str(e)}
                )

        # Build final system status
        system_status = SystemHealthStatus(
            overall_status=overall_status,
            health_score_avg=round(avg_score, 1),
            total_agents=len(agent_healths),
            ok_agents=sum(1 for h in agent_healths.values() if h.status == "OK"),
            warning_agents=sum(1 for h in agent_healths.values() if h.status == "WARNING"),
            critical_agents=sum(1 for h in agent_healths.values() if h.status == "CRITICAL"),
            down_agents=sum(1 for h in agent_healths.values() if h.status == "DOWN"),
            systemic_risk=systemic_risk,
            cascade_detected=bool(cascades),
            cascades=cascades,
            prioritized_actions=actions,
            issues=cascades,
            timestamp=datetime.now().isoformat()
        )

        self._next_step(
            reasoning,
            "Full system health check completed",
            output_data={
                "overall_status": overall_status,
                "total_steps": len(reasoning)
            }
        )

        logger.info(
            "System health check completed",
            extra={
                "status": overall_status,
                "score": avg_score,
                "systemic_risk": systemic_risk,
                "cascades": len(cascades)
            }
        )

        return {
            "system_health": asdict(system_status),
            "agent_healths": {name: asdict(health) for name, health in agent_healths.items()},
            "executive_summary": llm_summary,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("system_health")
    async def quick_ping(self) -> Dict[str, Any]:
        """
        Fast liveness check (from original Cloud9 design)

        Lightweight HTTP-only check without metrics or scoring
        """
        reasoning: List[ReasoningStep] = []
        self._next_step(reasoning, "Quick ping initiated")

        results = {}
        for name, url in self.KNOWN_AGENTS.items():
            results[name] = await self._ping_agent(name, url)

        healthy_count = sum(
            1 for r in results.values()
            if r.get("reachable") and r.get("healthy")
        )
        total = len(results)

        self._next_step(
            reasoning,
            "Quick ping completed",
            output_data={
                "healthy": healthy_count,
                "total": total,
                "status": "healthy" if healthy_count == total else "degraded"
            }
        )

        return {
            "system_status": "healthy" if healthy_count == total else "degraded",
            "healthy_agents": healthy_count,
            "total_agents": total,
            "agent_status": results,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("system_health")
    async def diagnose_agent(
            self,
            agent_name: str
    ) -> Dict[str, Any]:
        """
        Deep diagnostics for specific agent

        Provides detailed analysis with history and recommendations

        Args:
            agent_name: Name of agent to diagnose

        Returns:
            Detailed diagnostic report
        """
        reasoning: List[ReasoningStep] = []
        self._next_step(
            reasoning,
            "Agent diagnostics requested",
            input_data={"agent_name": agent_name}
        )

        if agent_name not in self.KNOWN_AGENTS:
            return {
                "error": f"Unknown agent: {agent_name}",
                "known_agents": list(self.KNOWN_AGENTS.keys())
            }

        # Get current metrics
        metrics = await self._get_metrics_from_agent()
        agent_metrics = metrics.get(agent_name, {
            "tasks_processed": 0,
            "errors": 0,
            "latency_ms": None
        })

        # Ping agent
        url = self.KNOWN_AGENTS[agent_name]
        ping_result = await self._ping_agent(agent_name, url)

        self._next_step(
            reasoning,
            "Collected agent data",
            output_data={
                "reachable": ping_result.get("reachable"),
                "tasks_processed": agent_metrics.get("tasks_processed"),
                "errors": agent_metrics.get("errors")
            }
        )

        # Compute health
        tasks = agent_metrics.get("tasks_processed", 0)
        errors = agent_metrics.get("errors", 0)
        error_rate = errors / max(tasks, 1)

        score = self._compute_health_score(
            error_rate=error_rate,
            latency_ms=agent_metrics.get("latency_ms"),
            tasks_processed=tasks,
            reachable=ping_result.get("reachable", False)
        )

        status = self._classify_status(score)
        trend, confidence = self._detect_trend(agent_name, score)

        # Build health status
        health = AgentHealthStatus(
            agent_name=agent_name,
            status=status,
            health_score=score,
            error_rate=error_rate,
            tasks_processed=tasks,
            errors=errors,
            response_time_ms=ping_result.get("response_time_ms") or agent_metrics.get("latency_ms"),
            reachable=ping_result.get("reachable", False),
            trend=trend,
            trend_confidence=confidence,
            issues=[],
            recommendations=[]
        )

        health.recommendations = self._generate_recommendations(health)

        self._next_step(
            reasoning,
            "Health assessment completed",
            output_data={
                "status": status,
                "score": score,
                "trend": trend
            }
        )

        # Get historical data
        history = self.agent_history.get(agent_name, [])

        # Check dependencies
        dependencies = self.DEPENDENCY_GRAPH.get(agent_name, [])
        dependents = [
            name for name, deps in self.DEPENDENCY_GRAPH.items()
            if agent_name in deps
        ]

        self._next_step(
            reasoning,
            "Diagnostics completed",
            output_data={
                "history_points": len(history),
                "dependencies": len(dependencies),
                "dependents": len(dependents)
            }
        )

        return {
            "agent_name": agent_name,
            "health": asdict(health),
            "history": history[-10:] if history else [],  # Last 10 points
            "dependencies": dependencies,
            "dependents": dependents,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("system_health")
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Lightweight system status for UI/dashboards

        Returns:
            Simplified system status
        """
        reasoning: List[ReasoningStep] = []
        self._next_step(reasoning, "Lightweight status check initiated")

        # Run full check but return simplified view
        full_result = await self.full_system_check()

        system = full_result["system_health"]
        actions = system["prioritized_actions"]

        # Get immediate actions only
        immediate = [a for a in actions if a.get("immediate")]

        self._next_step(
            reasoning,
            "Status summary prepared",
            output_data={
                "overall": system["overall_status"],
                "immediate_actions": len(immediate)
            }
        )

        return {
            "overall_status": system["overall_status"],
            "health_score": system["health_score_avg"],
            "systemic_risk": system["systemic_risk"],
            "ok_agents": system["ok_agents"],
            "warning_agents": system["warning_agents"],
            "critical_agents": system["critical_agents"],
            "down_agents": system["down_agents"],
            "immediate_actions": immediate[:3],  # Top 3 urgent
            "cascade_detected": system["cascade_detected"],
            "timestamp": system["timestamp"],
            "reasoning": reasoning
        }

    @log_method
    async def get_agent_history(
            self,
            agent_name: str,
            limit: int = 24
    ) -> Dict[str, Any]:
        """
        Get historical health data for agent

        Args:
            agent_name: Agent to query
            limit: Maximum number of historical points

        Returns:
            Historical health data with statistics
        """
        if agent_name not in self.KNOWN_AGENTS:
            return {
                "error": f"Unknown agent: {agent_name}",
                "known_agents": list(self.KNOWN_AGENTS.keys())
            }

        history = self.agent_history.get(agent_name, [])

        # Get most recent N points
        recent_history = history[-limit:] if len(history) > limit else history

        # Calculate statistics
        if recent_history:
            scores = [h["score"] for h in recent_history]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            # Detect overall trend
            if len(scores) >= 3:
                trend, confidence = self._detect_trend(agent_name, scores[-1])
            else:
                trend, confidence = "insufficient_data", 0.0
        else:
            avg_score = min_score = max_score = 0
            trend = "no_data"
            confidence = 0.0

        return {
            "agent_name": agent_name,
            "history": recent_history,
            "statistics": {
                "data_points": len(recent_history),
                "avg_score": round(avg_score, 1),
                "min_score": min_score,
                "max_score": max_score,
                "trend": trend,
                "trend_confidence": confidence
            }
        }

    @log_method
    async def analyze_anomalies(
            self,
            custom_status: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        AI-powered anomaly analysis (from original Cloud9 design)

        Args:
            custom_status: Optional custom status data to analyze

        Returns:
            LLM-powered anomaly analysis
        """
        reasoning: List[ReasoningStep] = []
        self._next_step(reasoning, "Anomaly analysis requested")

        # Get current status if not provided
        if custom_status:
            data = custom_status
            self._next_step(
                reasoning,
                "Using provided custom status",
                input_data={"source": "custom"}
            )
        else:
            full_check = await self.full_system_check()
            data = full_check["agent_healths"]
            self._next_step(
                reasoning,
                "Retrieved current system status",
                output_data={"agents": len(data)}
            )

        # Build analysis prompt
        prompt = f"""AI Esports Coaching System — Anomaly Detection Report

Current agent status:
{chr(10).join([f"• {name}: {info['status']} (score: {info['health_score']}, error_rate: {info['error_rate'] * 100:.1f}%, trend: {info['trend']})"
               for name, info in data.items()])}

As a senior DevOps engineer for this platform, analyze:
• Critical issues requiring immediate attention
• Performance anomalies and degradation patterns
• Potential cascading failures and dependencies
• Root cause hypotheses based on error patterns
• Recovery recommendations with priority order

Tone: Technical, urgent if needed, confident and actionable.
Keep analysis under 300 words."""

        try:
            analysis = await self.llm.chat(prompt)
            self._next_step(
                reasoning,
                "AI anomaly analysis completed",
                output_data={"analysis_length": len(analysis)}
            )
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            analysis = "AI analysis unavailable. Proceed with manual investigation."
            self._next_step(
                reasoning,
                "AI analysis failed",
                output_data={"error": str(e)}
            )

        return {
            "analysis": analysis,
            "input_data": data,
            "reasoning": reasoning
        }
