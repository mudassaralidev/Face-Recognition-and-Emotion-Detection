# Use the official Python image as the base image
FROM python:3.11.6

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies including libgl1-mesa-glx (libGL.so.1)
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Create an additional 'app' directory inside the '/app' directory in the container
RUN mkdir app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy your FastAPI application code into the container
COPY ./app /app/app

# Expose the port your FastAPI app will run on
EXPOSE 8000

# Define the command to run your FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
