# Use a lightweight Python base image
FROM python:3.12


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY added_requirements.txt .
# RUN pip install --no-cache-dir -r added_requirements.txt

ENV PYTHONUNBUFFERED=1
RUN pip install --upgrade sentence-transformers "huggingface-hub>=0.30.0,<1.0"

ENV HF_HOME=/app/.cache/huggingface
# Preload the embedding model so it is cached in the Docker image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en')"

COPY added_requirements.txt .
RUN pip install --no-cache-dir -r added_requirements.txt

# Copy the application code into the container
COPY app.py .
COPY .env .
COPY personas.json .

# Expose the port the Flask app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["echo", "Please specify a command in docker-compose.yml"]
