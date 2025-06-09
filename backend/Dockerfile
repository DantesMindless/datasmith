# Use the official Python image with a slim variant
FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DOCKER True

# Install dependencies
RUN pip install --upgrade pip
RUN pip install poetry

# Set work directory
WORKDIR /app

# Copy only the dependency files to leverage Docker cache
COPY pyproject.toml /app/

# Install dependencies
RUN poetry config virtualenvs.create false
RUN poetry install --no-root

# Copy project files and entrypoint.sh
COPY . /app/
COPY entrypoint.sh /entrypoint.sh

# Collect static files
RUN poetry run python manage.py collectstatic --noinput

# Make entrypoint executable
RUN chmod +x /entrypoint.sh

# Expose port 8000
EXPOSE 8000

# Set the entrypoint

ENTRYPOINT ["/entrypoint.sh"]
