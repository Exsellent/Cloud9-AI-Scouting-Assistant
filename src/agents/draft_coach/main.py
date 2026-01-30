from agents.draft_coach.agent import DraftCoachAgent

agent = DraftCoachAgent()
app = agent.app


@app.get("/")
async def root():
    return {
        "message": "Cloud9 AI Scouting Assistant — DraftCoach Agent is running",
        "agent": agent.name,
        "available_tools": list(agent.tools.keys()),
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8401)
