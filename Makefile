# Makefile
# Mark targets as not file-dependent; they are always executed
.PHONY: start startall down reset tar install help

dev:
	MODE=local uvicorn app.main:app --reload

start: 
	docker compose up -d

startall: 
	docker compose up --build -d

down:
	docker stop vergo_puppet

reset: down
	docker rm vergo_puppet

tar: 
	docker build -t vergo_puppet -f Dockerfile .
	docker save vergo_puppet -o vergo_puppet.tar

install:
	# Stop the container if it exists
	@if docker ps -a --format '{{.Names}}' | grep -q '^vergo_puppet$$'; then \
		docker stop vergo_puppet; \
		docker rm vergo_puppet; \
	else \
		echo "No running container to stop."; \
	fi
	# Remove the image if it exists
	@if docker images -q vergo_puppet; then \
		docker image rm vergo_puppet; \
	else \
		echo "No image to remove."; \
	fi
	# Load the image from the tarball
	docker load -i vergo_puppet.tar
	# Start the container with the production configuration
	docker compose -f docker-compose.prod.yml up -d

help:
	@echo ""
	@echo "~~ Vergo Apis Makefile ~~"
	@echo ""
	@echo "\033[33m make start\033[39m    : Démarre le projet"
	@echo "\033[33m make startall\033[39m : Build et démarre le projet"
	@echo "\033[33m make down\033[39m     : Stop le projet"
	@echo "\033[33m make reset\033[39m    : Reset les containers, les volumes, les networks et les données local"
	@echo ""