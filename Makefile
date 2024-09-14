start: 
	docker compose up -d

startall: 
	docker compose up --build -d

down:
	docker stop vergo_puppet

reset: down
	docker rm vergo_puppet

# Build the Docker image and save it as a tarball
tar: 
	docker build -t vergo_puppet -f Dockerfile .
	docker save vergo_puppet -o vergo_puppet.tar

# Install the Docker image by loading it from a tarball and running it
install:
	docker stop vergo_puppet
	docker rm vergo_puppet
	docker image rm vergo_puppet
	docker load -i vergo_puppet.tar
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