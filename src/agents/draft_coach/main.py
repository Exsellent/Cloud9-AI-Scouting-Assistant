from fastapi import FastAPI

from agents.draft_coach.agent import DraftCoachAgent
from shared.cors_patch import add_cors

agent = DraftCoachAgent()
app: FastAPI = agent.app
add_cors(agent.app)


@app.get("/")
async def root():
    return {
        "message": "Cloud9 AI Scouting Assistant — DraftCoach Agent is running!",
        "agent": agent.name,
        "available_tools": list(agent.tools.keys()),
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent.app, host="0.0.0.0", port=8401)
