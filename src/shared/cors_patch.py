from fastapi.middleware.cors import CORSMiddleware

def add_cors(app):
    """Add CORS middleware to FastAPI app"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )