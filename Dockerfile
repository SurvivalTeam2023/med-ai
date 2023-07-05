# Use the official Python base image with version 3.9
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port that FastAPI will be running on (by default, 8000)
EXPOSE 8000

# Command to start the application with Uvicorn
CMD [ "python","main.py" ]