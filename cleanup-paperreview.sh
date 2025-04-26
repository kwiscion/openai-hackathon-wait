#!/bin/bash

echo "Attempting to drop the paperreview database from PostgreSQL..."

# Check if container is running
if docker ps | grep -q paper_review_db; then
  echo "Container found. Attempting cleanup..."
  
  # Use the same approach that works in start-db.sh
  echo "Dropping paperreview database..."
  echo "DROP DATABASE IF EXISTS paperreview;" | docker exec -i paper_review_db psql -U postgres
  
  echo "Dropping paperreview role..."
  echo "DROP ROLE IF EXISTS paperreview;" | docker exec -i paper_review_db psql -U postgres
  
  echo "Cleanup process completed."
else
  echo "PostgreSQL container 'paper_review_db' is not running."
  echo "Start it first with: docker-compose up -d"
  echo "Or use './start-db.sh' to start the database properly."
fi 