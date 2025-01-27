from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from story_generator import modified_generate_story, store_feedback, init_search_system
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json
import datetime
import os
import socket

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://vibe-1-ec3i.onrender.com",
        "http://localhost:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoryRequest(BaseModel):
    prompt: str

class FeedbackRequest(BaseModel):
    story_id: str
    rating: int
    comment: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    print("Starting FastAPI server...")
    try:
        port = int(os.getenv("PORT", 10000))
        print(f"Attempting to bind to port {port}")
        if not bind_port(port):
            raise RuntimeError(f"Failed to bind to port {port}")
            
        # Initialize search system
        if not init_search_system():
            raise RuntimeError("Failed to initialize search system")
            
    except Exception as e:
        print(f"Startup error: {e}")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "port": "10000",
        "host": "0.0.0.0"
    }

@app.head("/")
async def head():
    return {"status": "ok"}

@app.post("/generate")
async def generate_story(request: StoryRequest):
    response = modified_generate_story(request.prompt)
    if response["status"] == "error":
        raise HTTPException(status_code=500, detail=response["message"])
    return response

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    response = store_feedback(feedback.dict())
    if response["status"] == "error":
        raise HTTPException(status_code=500, detail=response["message"])
    return response

def bind_port(port: int) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('0.0.0.0', port))
        sock.close()
        return True
    except Exception as e:
        print(f"Port {port} binding failed: {e}")
        return False

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
