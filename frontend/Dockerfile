# Dockerfile for Frontend (Streamlit UI - Final Working Version)

# 1. Use a stable, MODERN base image
FROM python:3.10-slim-bullseye 

# Set environment variables for Streamlit networking
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS="0.0.0.0"

# Copy the requirements file for installation
# CRITICAL: This assumes requirements.txt is in the frontend folder
COPY ./requirements.txt /tmp/requirements.txt

# Set the working directory inside the container
WORKDIR /app

# 2. Install Streamlit and networking dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 3. Copy the application code and assets
# CRITICAL: Copy the streamlit_app.py file from the frontend folder
COPY ./frontend/streamlit_app.py /app/streamlit_app.py 
COPY ./frontend/image.png /app/image.png

EXPOSE 8501

# 4. Define startup command (The correct structure for Streamlit)
ENTRYPOINT ["streamlit", "run"]

# CMD passes the arguments (the file name and the port configuration)
CMD ["streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]