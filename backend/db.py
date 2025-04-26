import json
import os
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Database connection parameters from environment variables
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

# SQLAlchemy async engine
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
ASYNC_DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine instances
engine = create_engine(DATABASE_URL)
async_engine = create_async_engine(ASYNC_DATABASE_URL)

# Create async session factory
async_session = sessionmaker(
    async_engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Define metadata for reflecting tables (optional)
metadata = MetaData()

# Helper function to get a session
async def get_session():
    """Get a SQLAlchemy async session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

async def create_task(session: AsyncSession, file_name: str, task_name: str) -> int:
    """Create a new task and return its ID."""
    # Use text for direct SQL
    result = await session.execute(
        text("INSERT INTO tasks (file_name, status, task_name, history) VALUES (:file_name, 'processing', :task_name, '[]') RETURNING id"),
        {"file_name": file_name, "task_name": task_name}
    )
    task_id = result.scalar_one()
    await session.commit()
    return task_id

async def update_task_status(session: AsyncSession, task_id: int, status: str, current_step: Optional[str] = None) -> None:
    """Update the status of a task."""
    if current_step:
        await session.execute(
            text("UPDATE tasks SET status = :status, current_step = :current_step WHERE id = :id"),
            {"status": status, "current_step": current_step, "id": task_id}
        )
    else:
        await session.execute(
            text("UPDATE tasks SET status = :status WHERE id = :id"),
            {"status": status, "id": task_id}
        )
    await session.commit()

async def set_task_result(session: AsyncSession, task_id: int, result: Dict[str, Any]) -> None:
    """Set the result data for a completed task."""
    # Convert result to JSON if needed
    result_json = json.dumps(result) if not isinstance(result, str) else result
    
    await session.execute(
        text("UPDATE tasks SET result = :result WHERE id = :id"),
        {"result": result_json, "id": task_id}
    )
    await session.commit()

async def get_task(session: AsyncSession, task_id: int) -> Optional[Dict[str, Any]]:
    """Get task information by ID."""
    result = await session.execute(
        text("SELECT id, file_name, status, task_name, current_step, history, result, created_at, updated_at FROM tasks WHERE id = :id"),
        {"id": task_id}
    )
    row = result.fetchone()
    if row:
        # Convert to dictionary with proper column names and handle JSON
        task = {
            "id": row[0],
            "file_name": row[1],
            "status": row[2],
            "task_name": row[3],
            "current_step": row[4],
            "history": row[5] if isinstance(row[5], list) else json.loads(row[5] if row[5] else '[]'),
            "result": row[6],
            "created_at": row[7],
            "updated_at": row[8]
        }
        # Parse result JSON if it exists and is a string
        if task["result"] and isinstance(task["result"], str):
            task["result"] = json.loads(task["result"])
        return task
    return None

async def add_task_history(session: AsyncSession, task_id: int, history_entry: Dict[str, Any]) -> None:
    """Add an entry to the task history stored in the tasks table."""
    # First, get the current history array
    result = await session.execute(
        text("SELECT history FROM tasks WHERE id = :id"),
        {"id": task_id}
    )
    row = result.fetchone()
    
    if not row:
        return
    
    # Parse existing history if needed
    current_history = row[0] if isinstance(row[0], list) else json.loads(row[0] if row[0] else '[]')
    
    # Add the new entry
    current_history.append(history_entry)
    
    # Update the history in the database
    await session.execute(
        text("UPDATE tasks SET history = :history, current_step = :current_step WHERE id = :id"),
        {
            "history": json.dumps(current_history), 
            "current_step": history_entry.get("step"), 
            "id": task_id
        }
    )
    await session.commit()

# For backward compatibility
async def get_task_history(session: AsyncSession, task_id: int) -> List[Dict[str, Any]]:
    """Get the history of a task from the tasks table."""
    result = await session.execute(
        text("SELECT history FROM tasks WHERE id = :id"),
        {"id": task_id}
    )
    row = result.fetchone()
    
    if not row or not row[0]:
        return []
    
    # Parse the history if needed
    history = row[0] if isinstance(row[0], list) else json.loads(row[0] if row[0] else '[]')
    return [{"id": i, "history": entry, "created_at": None} for i, entry in enumerate(history)]

# For backward compatibility with session auto-creation
async def create_task_with_session(file_name: str, task_name: str) -> int:
    """Create a new task and return its ID (with auto-session)."""
    async with async_session() as session:
        return await create_task(session, file_name, task_name)

async def update_task_status_with_session(task_id: int, status: str, current_step: Optional[str] = None) -> None:
    """Update the status of a task (with auto-session)."""
    async with async_session() as session:
        await update_task_status(session, task_id, status, current_step)

async def get_task_with_session(task_id: int) -> Optional[Dict[str, Any]]:
    """Get task information by ID (with auto-session)."""
    async with async_session() as session:
        return await get_task(session, task_id)

async def add_task_history_with_session(task_id: int, history_entry: Dict[str, Any]) -> None:
    """Add an entry to the task history (with auto-session)."""
    async with async_session() as session:
        await add_task_history(session, task_id, history_entry)

async def get_task_history_with_session(task_id: int) -> List[Dict[str, Any]]:
    """Get the history of a task (with auto-session)."""
    async with async_session() as session:
        return await get_task_history(session, task_id) 