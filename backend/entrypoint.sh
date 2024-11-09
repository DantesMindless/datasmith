#!/bin/sh

# Exit script on any error
set -e

# Check the environment variable
if [ "$ENV" = "development" ]; then
    echo "Running in development mode"
    # Run migrations in development
    poetry run python manage.py migrate
    # Start the Django development server
    exec poetry run python manage.py runserver 0.0.0.0:8000
else
    echo "Running in production mode"
    # Run migrations in production
    poetry run python manage.py migrate --noinput
    # Start Gunicorn in production
    exec poetry run gunicorn --bind 0.0.0.0:8000 datasmith.wsgi:application --workers 3
fi
