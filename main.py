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

def bind_port(port: int) -> bool:
    """Try to bind to the specified port to ensure it's available"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('0.0.0.0', port))
        sock.close()
        return True
    except Exception as e:
        print(f"Port {port} binding failed: {e}")
        return False

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

# Configure CORS - This is crucial for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://vibe-1-ec3i.onrender.com", # Your actual frontend URL on Render
        "http://localhost:5173",   # Local development
        "*",  # Allow all origins temporarily for debugging
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoryRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    query_text: str
    story_text: str
    rating: int
    feedback_text: Optional[str] = None

@app.post("/generate-story")
async def generate_story(request: StoryRequest):
    try:
        # Add error handling for empty queries
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        story = modified_generate_story(request.query)
        
        # Check if story generation was successful
        if not story:
            raise HTTPException(status_code=500, detail="Failed to generate story")
            
        return {"story": story}
    except Exception as e:
        # Log the error for debugging
        print(f"Error generating story: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/store-feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        # Validate rating range
        if feedback.rating < 1 or feedback.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        # Validate required fields are not empty    
        if not feedback.query_text.strip():
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
            
        if not feedback.story_text.strip():
            raise HTTPException(status_code=400, detail="Story text cannot be empty")
            
        success = store_feedback(
            query_text=feedback.query_text,
            story_text=feedback.story_text,
            rating=feedback.rating,
            feedback_text=feedback.feedback_text
        )
        
        if success:
            return {"message": "Feedback stored successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store feedback")
    except Exception as e:
        print(f"Error storing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    try:
        # Initialize system
        pinecone_instance, db = init_search_system()
        if None in (pinecone_instance, db):
            print("Warning: Some components failed to initialize")
    except Exception as e:
        print(f"Startup error: {e}")

@app.get("/health")
async def health_check():
    try:
        pinecone_instance, db = init_search_system()
        return {
            "status": "healthy",
            "port": "10000",
            "database": "connected" if db else "disconnected",
            "pinecone": "connected" if pinecone_instance else "disconnected"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    print("Starting FastAPI server...")
    try:
        PORT = 10000  # Force port 10000
        print(f"Attempting to bind to port {PORT}")
        
        # Direct uvicorn run
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=PORT,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        print(f"Error starting server: {e}")
