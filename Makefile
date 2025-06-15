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
	docker stop puppet

reset: down
	docker rm puppet

tar: 
	docker build -t puppet -f Dockerfile .
	docker save puppet -o puppet.tar

install:
	# Stop the container if it exists
	@if docker ps -a --format '{{.Names}}' | grep -q '^puppet$$'; then \
		docker stop puppet; \
		docker rm puppet; \
	else \
		echo "No running container to stop."; \
	fi
	# Remove the image if it exists
	@if docker images -q puppet; then \
		docker image rm puppet; \
	else \
		echo "No image to remove."; \
	fi
	# Load the image from the tarball
	docker load -i puppet.tar
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