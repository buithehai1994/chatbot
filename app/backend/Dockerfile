# Use Python 3.9-slim as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt from the backend folder to the container
COPY ../requirements.txt ./  

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory inside the app folder
# Assuming your FastAPI code is in the 'app' folder inside the 'backend' folder
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run FastAPI using Uvicorn with the correct entry point
CMD ["uvicorn", "app.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
