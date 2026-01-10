from agents.match_history.agent import MatchHistoryAgent

agent = MatchHistoryAgent()

app = agent.app


@app.get("/")
async def root():
    return {
        "message": "Cloud9 AI Scouting Assistant — MatchHistory Agent is running!",
        "agent": agent.name,
        "available_tools": list(agent.tools.keys()),
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent.app, host="0.0.0.0", port=8402)
