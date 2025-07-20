# Vergo Puppet

Vergo Puppet is an AI service for managing, training and running neural network models. It exposes REST APIs to create, train, test and use models for tasks ranging from regression to semantic search.

## Features

- Multiple neural network architectures (Simple NN, GRU, Siamese, LSTM, Embedding)
- Token-based authentication for all endpoints
- Docker support for production deployments
- Extensible design following a hexagonal architecture

## Documentation

* [Models](doc/models.md) – detailed information on available neural network types
* [API Guide](doc/api-guide.md) – example requests for each endpoint
* [LLaMA CUDA Setup](doc/llama-cuda.md) – compiling `llama-cpp-python` with GPU support
* [Troubleshooting](doc/troubleshooting.md) – common issues and solutions

## Requirements

- Python 3.9+
- `pip`
- MongoDB server
- Optional: Docker and Docker Compose

## Configuration

Environment variables are defined in a `.env` file. The most important settings are:

| Variable        | Description                                   | Default                                              |
|-----------------|-----------------------------------------------|------------------------------------------------------|
| `SECRET_KEY`    | Secret key used to sign JWT tokens            | `SECRET_KEY`                                         |
| `MODE`          | `dev`, `test` or `prod`                       | `prod`                                               |
| `MONGO_URI`     | MongoDB connection URI                        | `mongodb://root:password@localhost:27017`            |
| `MONGO_DB_NAME` | MongoDB database name                         | `puppet`                                             |
| `DEBUG`         | Enable debug logging (`true` or `false`)      | `false`                                              |

Example `.env` file:

```bash
SECRET_KEY=mySuperSecretKey
MODE=dev
MONGO_URI=mongodb://root:password@localhost:27017
MONGO_DB_NAME=puppet
DEBUG=true
```

## Installation

### Local Development

1. Clone the repository
   ```bash
git clone <repo-url>
cd vergo_puppet
```
2. (Optional) create a virtual environment
   ```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install dependencies
   ```bash
pip install -r requirements-dev.txt
pip install -r requirements-prod.txt
```
4. Start the API with hot reload
   ```bash
MODE=dev uvicorn app.main:app --reload
```
5. Open <http://localhost:8000/docs> to browse the Swagger UI.

### Docker

1. Adjust your `.env` with `MODE=prod` and production credentials.
2. Build and start containers
   ```bash
docker compose up --build -d
```
3. The API will be available on port `3004`. Visit <http://localhost:3004/docs>.

## Running Tests

The project uses `pytest`.

```bash
pytest
```

## API Overview

All requests require a Bearer token in the `Authorization` header. The API exposes endpoints to:

- create and train models
- search or test models
- manage embeddings
- store and query "things" (embedded objects)

Check the Swagger UI for full documentation of each route.

For detailed examples of every endpoint, see [doc/api-guide.md](doc/api-guide.md).

## Project Structure

```
app/
├─ apis/           # FastAPI route definitions
├─ services/       # Infrastructure services (DB, logging, ...)
├─ usecases/       # Business logic
├─ neural_network/ # Model implementations
└─ main.py         # FastAPI app
```

Other notable files:

- `Dockerfile` and `docker-compose.yml` for container deployment
- `requirements-dev.txt` and `requirements-prod.txt` for dependencies

## GPU Support

Instructions to compile `llama-cpp-python` with CUDA have been moved to [doc/llama-cuda.md](doc/llama-cuda.md).

## Troubleshooting

Common issues and solutions are listed in [doc/troubleshooting.md](doc/troubleshooting.md).

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes with clear commit messages.
3. Submit a pull request to the `main` branch.

Please follow the existing code style and add tests for any new functionality.
