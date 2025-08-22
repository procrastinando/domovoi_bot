# ---- Stage 1: The Builder ----
# This stage installs dependencies, including any that need to be compiled.
FROM python:3.12-alpine AS builder

# Install build-time system dependencies
RUN apk add --no-cache build-base

WORKDIR /app

# Copy only the requirements file first to leverage Docker's layer caching.
# This step will only be re-run if requirements.txt changes.
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: The Final Image ----
# This stage creates the clean, final image for production.
FROM python:3.12-alpine

# Install runtime system dependencies. ffmpeg is needed by the bot to process audio files.
RUN apk add --no-cache ffmpeg

WORKDIR /app

# Copy the installed packages from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy the application code from the build context (your git repo)
COPY app.py .
# COPY .env.example . # It's good practice to include an example env file
COPY language.yaml .

# Set the command to run your bot
CMD ["python", "app.py"]