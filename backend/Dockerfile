# backend/Dockerfile

# Use the official Python image with a slim variant
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies
RUN pip install --upgrade pip
RUN pip install poetry

# Set work directory
WORKDIR /app

# Copy only the dependency files to leverage Docker cache
COPY pyproject.toml poetry.lock /app/

# Install dependencies
RUN poetry config virtualenvs.create false
RUN poetry install --no-root

# Copy project
COPY . /app/

# Expose port 8000
EXPOSE 8000

# Run the Django development server
CMD ["poetry", "run", "python", "manage.py", "runserver", "0.0.0.0:8000"]
