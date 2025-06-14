#!/bin/sh
set -e

echo "Updating poetry.lock and installing dependencies..."
poetry lock
poetry install --no-root

if [ "$ENV" = "development" ]; then
    echo "Running in development mode"
    poetry run python manage.py migrate
    poetry run python manage.py create_superuser
fi

# No command passed → default to Django runserver or Gunicorn
if [ $# -eq 0 ]; then
    if [ "$ENV" = "development" ]; then
        echo "Starting Django development server"
        exec poetry run python manage.py runserver 0.0.0.0:8000
    else
        echo "Starting Gunicorn production server"
        exec poetry run gunicorn --bind 0.0.0.0:8000 datasmith.wsgi:application --workers 3
    fi
else
    echo "Running custom command: $@"
    exec "$@"
fi
