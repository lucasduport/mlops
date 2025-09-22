# TP2 - MLOps with MLflow and FastAPI

## Overview
This project demonstrates MLOps practices using MLflow for model tracking and FastAPI for model serving.

## What I Built

### 1. Model Training
- **Script**: `training/train.py`
- Trains a Logistic Regression model on the Iris dataset
- Logs experiments, parameters, and metrics to MLflow
- Registers the model in MLflow Model Registry

### 2. Model Serving
- **Normal Server**: `server/normal/main.py` - Standard model serving with FastAPI
- **Canary Server**: `server/canary/main.py` - Canary deployment implementation
- REST API endpoints for predictions and model updates

### 3. Infrastructure
- **MLflow UI**: Model tracking and experiment management
- **Docker Compose**: Containerized services (MLflow + FastAPI server)
- **FastAPI**: High-performance API for model inference

## Quick Start

1. **Start services**:
   ```bash
   docker compose up
   ```

2. **Train model**:
   ```bash
   python training/train.py
   ```

3. **Access services**:
   - MLflow UI: http://localhost:5000
   - API Server: http://localhost:8000

## Key Features
- Experiment tracking with MLflow
- Model versioning and registry
- RESTful API for model inference
- Containerized deployment
- Canary deployment strategy

## Tech Stack
- Python 3.11+
- MLflow
- FastAPI
- Scikit-learn
- Docker