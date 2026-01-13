from agents.stats_tracker.agent import StatsTrackerAgent

agent = StatsTrackerAgent()
app = agent.app


@app.get("/")
async def root():
    return {
        "message": "Cloud9 AI Scouting Assistant — StatsTracker Agent is running!",
        "agent": agent.name,
        "available_tools": list(agent.tools.keys()),
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent.app, host="0.0.0.0", port=8407)
