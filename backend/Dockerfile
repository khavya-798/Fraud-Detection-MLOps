# Dockerfile for Backend (FastAPI API - Final Working Version)

# 1. Use a stable, MODERN base image (Fixes the 404/Buster error)
FROM python:3.10-slim-bullseye 

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
# CRITICAL: This assumes requirements_api.txt is in the backend folder
COPY ./backend/requirements_api.txt /app/requirements.txt

# 2. Install all dependencies (FastAPI, XGBoost, Scikit-learn, etc.)
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3. Copy the application code and model assets
# We copy the necessary Python files
COPY ./backend/ /app/
# CRITICAL: Copy the favicon and the /models folder
COPY ./backend/favicon.png /app/favicon.png
COPY ./models /app/models/

# Expose the port (FastAPI default)
EXPOSE 8000

# 4. Define the startup command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]