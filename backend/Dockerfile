# Use the official Python image with a slim variant
FROM python:3.12-slim-bookworm

# Environment settings
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DOCKER True

# Install Poetry
RUN pip install --upgrade pip && pip install poetry

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* /app/

# Install dependencies
RUN poetry config virtualenvs.create false \
  && poetry install --no-root

# Copy project files
COPY . /app/

# Collect static files
RUN poetry run python manage.py collectstatic --noinput

# Add and make entrypoint executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

# Delegate behavior to entrypoint
ENTRYPOINT ["/entrypoint.sh"]
