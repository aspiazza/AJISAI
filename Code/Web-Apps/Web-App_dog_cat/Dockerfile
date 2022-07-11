# Use Python base image
FROM python:3.8.12-slim-buster

# Copy folders and files to app folder
COPY App /app/src
COPY requirements.txt /app

# Change work directory
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# Open port 80
EXPOSE 80

# Run these commands
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "80"]

# Run in terminal
# docker build -t ajisai-dogcat:0.1 .
# docker run -d --name dogcat-AI -p 80:80 ajisai-dogcat:0.1
