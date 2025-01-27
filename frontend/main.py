from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from story_generator import modified_generate_story, store_feedback
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json
import datetime

app = FastAPI()

# Configure CORS - Update with your frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default development port
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

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Add basic system checks
        return {
            "status": "healthy",
            "timestamp": str(datetime.datetime.now()),
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Health check failed")

if __name__ == "__main__":
    print("Starting FastAPI server...")
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="127.0.0.1",  # Using localhost
        port=port,
        workers=1,
        reload=False
    )