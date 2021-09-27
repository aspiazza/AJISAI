# Use Python base image
FROM python:3.8.12-slim-buster

# Copy folders and files to app folder
COPY ./App /app/src
COPY ./requirements.txt /app

# Change work directory
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# Open port 4000
EXPOSE 4000

# Run these commands
CMD ["uvicorn", "src.api:app", "--host=localhost", "--reload"]
