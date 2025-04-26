# Paper Review Application

A dockerized application for reviewing academic papers, consisting of a FastAPI backend, React frontend, and PostgreSQL database.

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Environment Variables

The application uses environment variables for configuration. You can customize them by:

1. Creating a `.env` file in the project root with the following variables:

```
# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=postgres

# Backend Configuration
ALLOWED_ORIGINS=http://localhost:5173

# Frontend Configuration
VITE_API_URL=http://localhost:8000
```

### Running the Application

1. Build and start the containers:

```bash
docker-compose up -d
```

2. Access the application:
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000

3. Stop the application:

```bash
docker-compose down
```

### Development Mode

The application is configured for development with hot-reloading:

- Frontend changes will automatically refresh the browser
- Backend changes will automatically restart the API server
- Volume mounts ensure local changes are reflected in the containers

## Architecture

- **Frontend**: React/TypeScript with Vite
- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL

## Containers

- **postgres**: Database service
- **backend**: FastAPI application
- **frontend**: React application

## Troubleshooting

- If you encounter database connection issues, ensure PostgreSQL has fully started before the backend attempts to connect.
- For permission issues with volumes, check Docker's volume permissions.
