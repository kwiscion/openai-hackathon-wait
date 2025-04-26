# Paper Review API Backend

This is a FastAPI application that handles paper review requests.

## Setup

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

Start the server with:
```bash
python main.py
```

Or directly with uvicorn:
```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

## API Documentation

Once the server is running, you can access the auto-generated API documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

- `GET /`: Health check endpoint
- `POST /upload`: Upload a PDF file for review 