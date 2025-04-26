from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import asyncio
import os
from datetime import datetime
from contextlib import asynccontextmanager

# Import database module and SQLAlchemy
import db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize connections on startup
    # Nothing needed here as SQLAlchemy creates connections on demand
    yield
    # Cleanup on shutdown: Dispose of the engine
    await db.async_engine.dispose()

app = FastAPI(title="Paper Review API", lifespan=lifespan)

# Enable CORS for the frontend
# Get allowed origins from environment or use defaults
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080,http://localhost:8081,http://127.0.0.1:8081,http://frontend:5173,http://frontend:8080"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SectionResult(BaseModel):
    title: str
    score: float
    feedback: str

class ReviewResult(BaseModel):
    title: str
    feedback: str
    sections: Optional[List[SectionResult]] = None

class TaskHistory(BaseModel):
    step: str
    status: str
    message: str
    timestamp: str

class TaskStatus(BaseModel):
    task_id: int
    status: str
    file_name: str
    current_step: Optional[str] = None
    history: Optional[List[TaskHistory]] = None
    data: Optional[Dict[str, Any]] = None

# Dependency for SQLAlchemy session
async def get_db():
    """Get a database session."""
    async for session in db.get_session():
        yield session

@app.get("/")
async def root():
    return {"message": "Paper Review API"}

@app.post("/upload", response_model=TaskStatus)
async def upload_pdf(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = None,
    session: AsyncSession = Depends(get_db)
):
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        
        # Check if the file is a PDF
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Create a new task in the database
        task_id = await db.create_task(session, file.filename, "Paper Review")
        print(f"Created new task with ID: {task_id}")
        
        # Add the first history entry
        first_history_entry = {
            "step": "upload",
            "status": "completed",
            "message": "File uploaded successfully",
            "timestamp": datetime.now().isoformat()
        }
        await db.add_task_history(session, task_id, first_history_entry)
        
        # Explicitly set the initial status and current_step
        # Use gathering_papers as the current_step to ensure proper display
        await db.update_task_status(session, task_id, "processing", "gathering_papers")
        
        # Start a background task to process the file
        if background_tasks is None:
            raise HTTPException(status_code=500, detail="Background tasks service unavailable")
            
        background_tasks.add_task(process_pdf_task, task_id)
        
        # Return task information
        task = await db.get_task(session, task_id)
        
        if not task:
            raise HTTPException(status_code=500, detail="Failed to create task")
        
        # Extract history from task for the response
        history_items = [TaskHistory(**item) for item in task.get("history", [])]
        
        return TaskStatus(
            task_id=task_id,
            status="processing",
            file_name=file.filename,
            current_step="gathering_papers",
            history=history_items
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error and return a 500 error
        print(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Server error during file upload: {str(e)}"
        )

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_job_status(
    task_id: int,
    session: AsyncSession = Depends(get_db)
):
    try:
        # Get task from database
        task = await db.get_task(session, task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found with ID: {task_id}")
        
        # Extract history items directly from the task
        history_items = []
        if task.get("history"):
            try:
                history_items = [TaskHistory(**item) for item in task.get("history", [])]
            except Exception as e:
                print(f"Error parsing task history: {str(e)}")
                # Continue with empty history rather than failing
        
        response = TaskStatus(
            task_id=task_id,
            status=task["status"],
            file_name=task["file_name"],
            current_step=task["current_step"],
            history=history_items
        )
        
        # If task is completed, add the review results from the result field
        if task["status"] == "completed" and task.get("result"):
            response.data = task["result"]
        
        # If task failed, include error info
        if task["status"] == "failed":
            response.data = {"error": True, "message": "Task processing failed"}
        
        return response
    except HTTPException:
        # Re-raise HTTP exceptions to maintain status codes
        raise
    except Exception as e:
        # Log the error and return a 500 error
        print(f"Error fetching task status: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Server error while fetching task status: {str(e)}"
        )

async def process_pdf_task(task_id: int):
    """Simulates the processing of a PDF file with multiple steps."""
    print(f"Starting task processing for task_id: {task_id}")
    # For background tasks we need to create our own session
    try:
        async with db.async_session() as session:
            steps = [
                {
                    "name": "gathering_papers",
                    "title": "Gathering Relevant Papers",
                    "duration": 2,  # seconds (reduced from 5)
                    "completion_message": "Found 15 relevant papers for comparison"
                },
                {
                    "name": "setting_up_reviews",
                    "title": "Setting Up Review Framework",
                    "duration": 3,  # seconds (reduced from 6)
                    "completion_message": "Review parameters configured successfully"
                },
                {
                    "name": "generating_reviews",
                    "title": "Generating Section Reviews",
                    "duration": 4,  # seconds (reduced from 8)
                    "completion_message": "Generated reviews for 5 sections"
                },
                {
                    "name": "meta_review",
                    "title": "Compiling Final Assessment",
                    "duration": 3,  # seconds (reduced from 7)
                    "completion_message": "Final review compiled successfully"
                }
            ]
            
            # First step - get current status to ensure it's properly set up
            currentTask = await db.get_task(session, task_id)
            if currentTask:
                print(f"Task {task_id} current step before processing: {currentTask.get('current_step')}")
                
                # Force set to gathering_papers as the first step
                if currentTask.get('current_step') != 'gathering_papers':
                    await db.update_task_status(session, task_id, "processing", "gathering_papers")
                    print(f"Task {task_id}: Force updated initial step to gathering_papers")
            
            # Process each step
            for step in steps:
                print(f"Task {task_id}: Starting step '{step['name']}'")
                
                # Start step - set step as in_progress
                await db.add_task_history(session, task_id, {
                    "step": step["name"],
                    "status": "in_progress",
                    "message": f"{step['title']}",
                    "timestamp": datetime.now().isoformat()
                })
                
                # CRITICAL: Update the task status to ensure current_step is set
                # This is what the frontend uses to update the highlighted step
                print(f"Task {task_id}: Setting current_step to '{step['name']}' (in_progress)")
                await db.update_task_status(session, task_id, "processing", step["name"])
                
                # Simulate processing
                await asyncio.sleep(step["duration"])
                
                # Complete step
                await db.add_task_history(session, task_id, {
                    "step": step["name"],
                    "status": "completed",
                    "message": step["completion_message"],
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update task status after completion
                # The current_step remains the same until the next step starts
                print(f"Task {task_id}: Step '{step['name']}' completed (current_step remains '{step['name']}')")
                await db.update_task_status(session, task_id, "processing", step["name"])
            
            # Prepare mock result response
            mock_result = {
                "success": True,
                "data": {
                    "title": "Research Paper Analysis",
                    "feedback": """
Abstract:
The abstract effectively summarizes the main points of the research, but could be more concise. Consider reducing it by 20% while maintaining key findings.

Methodology:
- Well-structured research design
- Clear explanation of data collection methods
- Sample size is appropriate for the study
- Statistical analysis methods are well-justified

Results:
The results are presented clearly with good use of data visualization. However, some graphs could benefit from better labeling.

Discussion:
Strong analysis of findings, but consider addressing the following limitations:
- Potential sampling bias
- External validity considerations
- Impact of time constraints on data collection

References:
All citations follow the required format. Consider including more recent sources (past 3 years) to strengthen the literature review.

Overall Score: 8.5/10
""",
                    "sections": [
                        {
                            "title": "Abstract",
                            "score": 7.5,
                            "feedback": "The abstract effectively summarizes the main points of the research, but could be more concise."
                        },
                        {
                            "title": "Methodology",
                            "score": 9.0,
                            "feedback": "Well-structured research design with clear explanation of data collection methods."
                        },
                        {
                            "title": "Results",
                            "score": 8.5,
                            "feedback": "Results are presented clearly with good use of data visualization. Some graphs could benefit from better labeling."
                        },
                        {
                            "title": "Discussion",
                            "score": 8.0,
                            "feedback": "Strong analysis of findings with some limitations to address."
                        },
                        {
                            "title": "References",
                            "score": 9.5,
                            "feedback": "All citations follow the required format. Consider including more recent sources."
                        }
                    ]
                }
            }
            
            # Store the result in the database
            await db.set_task_result(session, task_id, mock_result)
            
            # Mark task as completed
            await db.update_task_status(session, task_id, "completed", "meta_review")
    except Exception as e:
        # Log the error
        print(f"Error processing task {task_id}: {str(e)}")
        
        # Attempt to update the task status to failed
        try:
            async with db.async_session() as session:
                await db.update_task_status(session, task_id, "failed")
                await db.add_task_history(session, task_id, {
                    "step": "error",
                    "status": "failed",
                    "message": f"An error occurred: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as inner_e:
            print(f"Failed to update task status to failed: {str(inner_e)}")

@app.get("/reset", response_model=Dict[str, str])
async def reset_tasks(session: AsyncSession = Depends(get_db)):
    """Debug endpoint to reset all tasks - use only in development"""
    try:
        await session.execute(text("DELETE FROM tasks"))
        await session.commit()
        return {"status": "success", "message": "All tasks deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting tasks: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
