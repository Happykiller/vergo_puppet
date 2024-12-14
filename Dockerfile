# Use a Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file first (for caching purposes)
COPY requirements-prod.txt /app/requirements-prod.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Download the spaCy model
RUN python -m spacy download fr_core_news_md

# Copy the rest of the project files
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
