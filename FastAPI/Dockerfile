FROM python:3.11-slim

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy code and install dependencies
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Launch FastAPI via Uvicorn
CMD uvicorn app:app --host=0.0.0.0 --port=$PORT
