import inspect
import json
import logging
import os
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


class MCPRequest(BaseModel):
    method: str
    params: dict
    id: int | None = None


def remove_nulls(obj):
    """
    Recursively remove None/null values from dicts, lists and Pydantic models
    This makes JSON responses cleaner in the UI
    """
    if isinstance(obj, BaseModel):
        # Convert model → dict without None
        return remove_nulls(obj.model_dump(exclude_none=True))

    if isinstance(obj, dict):
        return {
            k: remove_nulls(v)
            for k, v in obj.items()
            if v is not None
        }

    if isinstance(obj, list):
        return [remove_nulls(item) for item in obj if item is not None]

    return obj


class MCPAgent:
    def __init__(self, name: str):
        self.name = name
        self.app = FastAPI()
        self.tools: Dict[str, Any] = {}

        # Enable CORS for web UI
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/health")
        def health():
            return {"status": "ok", "agent": self.name}

        @self.app.get("/")
        def root():
            return {
                "message": "Cloud9 AI Scouting Agent",
                "agent": self.name,
                "available_tools": list(self.tools.keys()),
                "docs": "/docs"
            }

        @self.app.post("/mcp")
        async def mcp(request: Request):
            try:
                # Get raw body
                body = await request.body()
                # Parse JSON
                data = json.loads(body.decode('utf-8'))
                # Validate with Pydantic
                req = MCPRequest(**data)
            except json.JSONDecodeError as e:
                return {
                    "error": "Invalid JSON",
                    "details": str(e),
                    "hint": "Send valid JSON with 'method' and 'params'"
                }
            except ValidationError as e:
                return {
                    "error": "Invalid MCP request format",
                    "details": str(e),
                    "hint": "Required fields: method (str), params (dict)"
                }
            except Exception as e:
                return {
                    "error": "Request processing failed",
                    "details": str(e)
                }

            tool_name = req.method.replace("tools/", "")
            if tool_name not in self.tools:
                return {
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": list(self.tools.keys())
                }

            handler = self.tools[tool_name]

            try:
                # Execute handler (async or sync)
                if inspect.iscoroutinefunction(handler):
                    result = await handler(**req.params)
                else:
                    result = handler(**req.params)

                # Auto-fill agent field in reasoning steps
                if isinstance(result, dict) and "reasoning" in result:
                    for step in result["reasoning"]:
                        if isinstance(step, dict):
                            step.setdefault("agent", self.name)
                        elif hasattr(step, "agent"):
                            if getattr(step, "agent", None) is None:
                                step.agent = self.name

                # ✅ Remove all null values for cleaner UI display
                result = remove_nulls(result)

                return result

            except TypeError as e:
                # Better error message for parameter mismatches
                sig = inspect.signature(handler)
                expected_params = [p for p in sig.parameters.keys() if p != "self"]
                return {
                    "error": f"Invalid parameters for tool '{tool_name}'",
                    "details": str(e),
                    "received_params": list(req.params.keys()),
                    "expected_params": expected_params
                }
            except Exception as e:
                return {
                    "error": f"Tool execution failed: {tool_name}",
                    "details": str(e)
                }

    def register_tool(self, name: str, handler):
        self.tools[name] = handler
        logging.getLogger(self.name).info(f"Registered tool: {name}")
