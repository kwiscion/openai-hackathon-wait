#!/bin/bash

# Stop and remove existing container
echo "Stopping and removing existing database container..."
docker-compose down

# Remove the volume to clear existing data
echo "Removing database volume to clear all data..."
# Using correct volume name for postgres
docker volume rm openai-hackathon-wait_postgres_data || true

# Force remove the container if it still exists
echo "Removing container if it exists..."
docker rm paper_review_db --force || true

# Force remove the PostgreSQL image if it exists
echo "Removing PostgreSQL Docker image..."
docker rmi postgres:16 --force || true

# Pull the latest PostgreSQL image
echo "Pulling fresh PostgreSQL image..."
docker pull postgres:16

# Start a fresh PostgreSQL instance with updated schema
echo "Starting fresh PostgreSQL database with new schema..."
docker-compose up -d

# Wait for the database to be ready
echo "Waiting for database to be ready..."
sleep 5  # Give it some time to start up

# Show container status
echo "Container status:"
docker-compose ps

echo "Database has been reset with the new schema."
echo "Database connection details:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  User: postgres"
echo "  Password: postgres"
echo "  Database: postgres" 