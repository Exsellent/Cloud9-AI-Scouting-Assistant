from agents.counter_play.agent import CounterPlayAgent

agent = CounterPlayAgent()
app = agent.app


@app.get("/")
async def root():
    return {
        "message": "Cloud9 AI Scouting Assistant — CounterPlay Agent is running!",
        "agent": agent.name,
        "available_tools": list(agent.tools.keys()),
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent.app, host="0.0.0.0", port=8403)
