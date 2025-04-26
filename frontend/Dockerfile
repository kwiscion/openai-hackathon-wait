FROM oven/bun:1 as base

WORKDIR /app

# First copy the package files
COPY package.json bun.lockb ./

# Install dependencies
RUN bun install --frozen-lockfile

# Copy the rest of the application
COPY . .

# Set environment for development
ENV NODE_ENV=development
ENV VITE_API_URL=http://localhost:8000

# Expose the dev server port
EXPOSE 8081

# Start the application
CMD ["bun", "run", "dev", "--host", "0.0.0.0", "--port", "8081"] 