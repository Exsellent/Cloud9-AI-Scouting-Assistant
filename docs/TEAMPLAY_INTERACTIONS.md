```markdown
# 🔄 Agent Interaction Patterns

---

# ✅ Architecture: 6 Specialized Esports AI Agents + System Monitor

### ✅ Complete List of Agents

| № | Agent          | Port | Purpose                                                                           |
|---|----------------|------|-----------------------------------------------------------------------------------|
| 1 | DraftCoach     | 8401 | AI Drafting Assistant (LoL) — provides pick/ban recommendations and counter-picks |
| 2 | MatchHistory   | 8402 | Analyzes past matches and evaluates the current form of the team                  |
| 3 | CounterPlay    | 8403 | Identifies opponent weaknesses and suggests counter-strategies                    |
| 4 | ScoutingReport | 8404 | Main generator of automated scouting reports                                      |
| 5 | StatsTracker   | 8407 | Tracks statistics, win rates, tendencies, and player performance                  |
| 6 | SystemHealth   | 8406 | Monitors the health of all agents (self-healing & reliability layer)              |

_____

# 🔄 Agent Interaction Patterns

## 1. Basic Scouting Flow
```
User / Coach
  ↓
ScoutingReport (8404)
  ├─► GRIDClient → fetch recent matches (Valorant/LoL)
  └─► Groq LLM → generate coach-friendly report
  ↓
Final Scouting Report (key insights, strengths/weaknesses, counter-strategy)
```

## 2. Advanced Multi-Agent Analysis Flow
```
Coach → n8n Orchestrator (or direct API)
  ├─► ScoutingReport (8404) → main opponent breakdown
  ├─► MatchHistory (8402) → current form & trends
  ├─► CounterPlay (8403) → exploitable weaknesses & punishment tactics
  ├─► StatsTracker (8407) → team & player performance metrics
  └─► (optional) DraftCoach (8401) → LoL draft recommendations
          ↓
Aggregated Intelligence Report (combined insights from all agents)
```

## 3. Draft Strategy Flow (LoL only)
```
Coach → DraftCoach (8401)
  ├─► GRIDClient → fetch recent LoL matches
  ├─► analyze_opponent_pool → champion pools & win rates
  └─► recommend_draft → priority bans, picks, synergies, win conditions
  ↓
Professional LoL Draft Plan
```

## 4. System Reliability Flow (Continuous Background)
```
SystemHealth (8406) — runs periodically or on-demand
  ├─► /health ping → DraftCoach (8401)
  ├─► /health ping → MatchHistory (8402)
  ├─► /health ping → CounterPlay (8403)
  ├─► /health ping → ScoutingReport (8404)
  ├─► /health ping → StatsTracker (8407)
  └─► (if degraded) Groq LLM → anomaly analysis & recovery recommendations
          ↓
System Status: healthy / degraded + actionable summary
```

## 5. Full End-to-End Demo Workflow (via n8n)
```
Coach Trigger (n8n)
  ├─► Parallel calls:
  │    ├─► MatchHistory (8402) → form evaluation
  │    ├─► CounterPlay (8403) → counter-strategies
  │    ├─► StatsTracker (8407) → performance stats
  │    └─► DraftCoach (8401) → draft recommendations (LoL)
  └─► ScoutingReport (8404) → aggregates all data → final report
          ↓
Complete AI-Powered Scouting Package
```

---

These patterns demonstrate the **modular, scalable, and reliable** nature of the platform:
- Direct single-agent calls for quick insights
- Orchestrated multi-agent flows for deep analysis
- Built-in health monitoring for production stability
- GRID-powered real data with seamless demo fallback

The system is fully MCP-compliant and ready for real-world esports coaching.
```

```text
┌────────────────────────────────────────────────────────────────────┐
│                    n8n Orchestrator (Optional)                     │
│                (Workflow Automation & External Triggers)           │
└────────────┬────────────────┬──────────────── ┬────────────────────┘
             │                │                 │
       ┌─────▼─────┐    ┌─────▼───────┐    ┌─────▼───── ┐
       │ DraftCoach│    │ MatchHistory│    │ CounterPlay│
       │   :8401   │    │   :8402     │    │   :8403    │
       └────┬──────┘    └────┬────────┘    └─────┬──────┘
            │                │                   │
       ┌────▼─────┐    ┌─────▼────────┐    ┌─────▼────────┐
       │Scouting  │    │ StatsTracker │    │ SystemHealth │
       │Report    │    │    :8407     │    │    :8406     │
       │  :8404   │    └────┬─────────┘    └────┬─────────┘
       └────┬─────┘         │                   │
            │               │                   │
            └───────────────┴───────────────────┘
                            │
                ┌───────────▼───────────┐
                │     Shared Layer      │
                │-----------------------│
                │ • MCP Protocol (/mcp) │
                │ • LLM Client (Groq)   │
                │ • GRID Esports API    │
                │ • Reasoning Engine    │
                │ • Metrics & Logging   │
                │ • Error Handling      │
                └───────────────────────┘
```