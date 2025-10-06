End-to-End Credit Card Fraud Detection with MLOps

https://fraud-detection-mlops.streamlit.app/  Streamlit Cloud link!

🚀 Project Overview
This project demonstrates a full end-to-end MLOps workflow for a real-time credit card fraud detection system. It includes a trained XGBoost model, a backend API to serve the model, a frontend user interface for interaction, and is fully containerized with Docker for reproducibility and deployment.

The application allows a user to input transaction details via a web interface and receive an instant prediction on whether the transaction is fraudulent or legitimate.

Key Features:
Machine Learning Model: Utilizes a high-performance XGBoost classifier trained on an anonymized transaction dataset from Kaggle.

Real-Time API: A robust backend API built with FastAPI serves the ML model, handling data validation and prediction requests.

Interactive UI: A user-friendly frontend built with Streamlit provides sliders and input fields for all 30 model features.

Containerized Environment: The entire application (backend and frontend) is containerized using Docker and orchestrated with Docker Compose, ensuring a consistent and isolated environment.

Large File Handling: Correctly manages large dataset files using Git LFS (Large File Storage).

🛠️ Tech Stack
Machine Learning: Python, Pandas, Scikit-learn, XGBoost, Joblib

Backend: FastAPI, Uvicorn

Frontend: Streamlit

Infrastructure & MLOps: Docker, Docker Compose

Version Control: Git & Git LFS

⚙️ Local Setup and Installation
To run this project on your local machine, please follow these steps.

Prerequisites
Docker installed on your machine.

Git installed.

Git LFS installed.

Installation Steps
Clone the repository:

git clone [https://github.com/khavya-798/Fraud-Detection-MLOps.git](https://github.com/khavya-798/Fraud-Detection-MLOps.git)
cd Fraud-Detection-MLOps

Pull LFS files:
Ensure you have the large dataset file by pulling the LFS data.

git lfs pull

Build and Run with Docker Compose:
This single command will build the Docker images for both the frontend and backend, and start the services.

docker-compose up --build

Access the Application:

Frontend (Streamlit App): Open your browser and go to http://localhost:8501

Backend (API Docs): You can access the API documentation at http://localhost:8000/docs

📂 Project Structure
Fraud-Detection-MLOps/
├── .gitignore
├── README.md
├── backend/
│   ├── app.py             # FastAPI application
│   ├── Dockerfile         # Docker instructions for backend
│   └── ...
├── data/
│   └── creditcard.csv     # Dataset (tracked by Git LFS)
├── frontend/
│   ├── streamlit_app.py   # Streamlit application
│   ├── Dockerfile         # Docker instructions for frontend
│   └── image.png
├── models/
│   ├── credit_fraud.pkl   # Trained XGBoost model
│   └── scaler.pkl         # Fitted StandardScaler
└── docker-compose.yml     # Docker Compose orchestration file
