# Makefile for Flask + Vue.js Full-Stack Application in Development Containers
#
# Targets:
#   start            Start the development containers using Docker Compose.
#   db-upgrade       Apply the latest database schema migrations.
#   migrate          Generate a new database migration.
#   drop-all         Stop and remove all containers, networks, and volumes, and prune unused data.
#   drop             Stop and remove all containers without pruning.
#   seed             Seed the database with initial data.
#   units            Run all backend tests.
#   unit             Run unit tests for a specific test file. usage: make unit f=<test_file>
#   shell            Run a shell in the API container.
#   test-backend     Run all backend tests.
#   help             Display Makefile targets.
# Define variables
DOCKER_COMPOSE_FILE := docker-compose.devcontainer.yml
ENV_FILE := .env.dev

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    API_CONTAINER := datasmith_django_1
else ifeq ($(UNAME_S),Darwin) # macOS
    API_CONTAINER := datasmith-django-1
else # Assume Windows for other cases
    API_CONTAINER := datasmith-django-1
endif

UI_CONTAINER := deep-web-ui-1
# Targets
.PHONY: start
start: ## Start the development containers using Docker Compose.
	docker-compose up -d
.PHONY: start-db
start-db: ## Start the development containers using Docker Compose.
	docker-compose -f=docker-compose-db.yml up -d
.PHONY: stop
stop: ## Start the development containers using Docker Compose.
	docker-compose stop
.PHONY: down
down: ## Start the development containers using Docker Compose.
	docker-compose down
	docker system prune -a -f
	docker volume prune -f
.PHONY: migrate
migrate: ## Upgrade the database schema to the latest revision.
	docker exec -it $(API_CONTAINER) poetry run python manage.py migrate
.PHONY: makemigrations
makemigrations: ## Upgrade the database schema to the latest revision.
	docker exec -it $(API_CONTAINER) poetry run python manage.py makemigrations
.PHONY: test
test: ## Upgrade the database schema to the latest revision.
	docker exec -it $(API_CONTAINER) poetry run python manage.py test
.PHONY: requirements
requirements: ## Upgrade the database schema to the latest revision.
	docker exec -it $(API_CONTAINER) poetry export --without-hashes --format=requirements.txt > requirements.txt
.PHONY: lint
lint: ## lint backend code
	ruff check . --fix
.PHONY: superuser
superuser: ## lint backend code
	docker exec -it $(API_CONTAINER) poetry run python manage.py create_superuser
