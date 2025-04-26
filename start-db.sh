#!/bin/bash

# Check for reset flag
RESET_DB=false
for arg in "$@"; do
  if [ "$arg" == "--reset" ]; then
    RESET_DB=true
  fi
done

# If reset flag is provided, stop container and remove volume first
if [ "$RESET_DB" == "true" ]; then
  echo "Reset flag detected. Stopping existing container and cleaning up..."
  docker-compose down
  
  # Remove the volume to clear existing data
  echo "Removing database volume to clear all data..."
  VOLUME_NAME=$(basename "$PWD")_postgres_data
  echo "Volume name identified as: $VOLUME_NAME"
  docker volume rm $VOLUME_NAME || true
  
  # Force remove the container if it still exists
  echo "Removing container if it exists..."
  docker rm paper_review_db --force || true
  
  echo "Cleanup complete."
fi

# Start the PostgreSQL database using Docker Compose
echo "Starting PostgreSQL database..."
docker-compose up -d

# Wait for the database to be ready
echo "Waiting for database to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

# Check if PostgreSQL is accepting connections
until docker exec paper_review_db pg_isready -U postgres > /dev/null 2>&1; do
  RETRY_COUNT=$((RETRY_COUNT + 1))
  if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Failed to connect to PostgreSQL after $MAX_RETRIES attempts. Exiting."
    exit 1
  fi
  echo "PostgreSQL is not ready yet. Waiting..."
  sleep 1
done

echo "PostgreSQL is now accepting connections!"

# Show container status
echo "Container status:"
docker-compose ps

echo "Creating tables in the default 'postgres' database..."

# SQL schema to create the tasks table and triggers
read -r -d '' SQL_SCHEMA << 'EOF'
-- Drop existing tables and functions if they exist
DROP TRIGGER IF EXISTS update_tasks_updated_at ON tasks;
DROP FUNCTION IF EXISTS update_updated_at_column();
DROP TABLE IF EXISTS tasks CASCADE;
DROP TABLE IF EXISTS task_history CASCADE;

-- Drop the paperreview database if it exists (to clean up cached database)
DROP DATABASE IF EXISTS paperreview;

-- Create tasks table
CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    task_name TEXT NOT NULL,
    current_step TEXT,
    history JSONB DEFAULT '[]',
    result JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create a function to update the updated_at timestamp
CREATE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create a trigger to automatically update the updated_at column in tasks table
CREATE TRIGGER update_tasks_updated_at
BEFORE UPDATE ON tasks
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Confirm completion
SELECT 'Schema initialization complete: Tables created in the postgres database' AS status;
EOF

# Execute the SQL schema using docker exec to connect to the default postgres database
echo "$SQL_SCHEMA" | docker exec -i paper_review_db psql -U postgres

echo "Tables created successfully in the postgres database."
echo "Database is ready. Backend can now connect to it."
echo "Database connection details:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  User: postgres"
echo "  Password: postgres"
echo "  Database: postgres (tables have been created here)"
echo ""
echo "If you want to reset the database completely, run: ./start-db.sh --reset" 