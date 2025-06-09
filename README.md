# Vergo Puppet

AI service for model management, training, and predictions.  
This project supports multiple neural network architectures (Simple NN, GRU, SIAMESE, LSTM, EMBEDDING) to tackle various tasks including regression, classification, similarity checks, and time-series forecasting. It also provides embedding-based search capabilities for storing and querying user data ("things").

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Configuration](#configuration)
4. [Installation](#installation)
    - [Local Installation (Development)](#local-installation-development)
    - [Docker Installation (Production)](#docker-installation-production)
5. [Development Server](#development-server)
6. [Running Tests](#running-tests)
7. [APIs Overview](#apis-overview)
8. [Model Details](#model-details)
    - [Puppet-o1 (Simple Model)](#puppet-o1-simple-model)
    - [Puppet-o2 (GRU Model)](#puppet-o2-gru-model)
    - [Puppet-o3 (SIAMESE Model)](#puppet-o3-siamese-model)
    - [Puppet-o4 (LSTM Model)](#puppet-o4-lstm-model)
    - [Puppet-o5 (Embedding Model)](#puppet-o5-embedding-model)
9. [Project Structure](#project-structure)
10. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)

---

## Introduction

**Vergo Puppet** is an AI-powered service designed for efficient model management, training, and inference tasks. It supports multiple neural network architectures (Simple NN, GRU, SIAMESE, LSTM) to cover a wide range of use cases:  
- Regression tasks (house price estimation, etc.)  
- Classification tasks (labeling sequences, etc.)  
- Similarity checks (semantic similarity, etc.)  
- Time-series forecasting (meteorological data, etc.)

### Key Features
- **Modular neural network architecture**: Easily swap in or out different model types.
- **RESTful APIs**: Endpoints for creating, training, testing, and searching within models.
- **Flexible deployment**: Designed to run either as a local development setup or inside Docker for production.
- **Data caching**: Allows precomputing vectors for faster similarity searches.
- **Universal embedding model**: Train and use embeddings for semantic search and "thing" indexing.

---

## Prerequisites

- **Python 3.9+**  
- **pip** (latest version recommended)  
- **Docker** and **Docker Compose** (for containerized deployment)  
- Basic familiarity with using virtual environments (recommended but not strictly required).

---

## Configuration

Vergo Puppet uses a `.env` file (or a custom environment file) to configure environment variables:

| Variable      | Description                                      | Default                             |
|---------------|--------------------------------------------------|-------------------------------------|
| `SECRET_KEY`  | Secret key for JWT token generation and validation | `SECRET_KEY`                        |
| `MODE`        | Execution mode (`dev`, `test`, or `prod`)          | `prod`                              |
| `MONGO_URI`   | Connection string for MongoDB                      | `mongodb://root:password@localhost:27017` |
| `MONGO_DB_NAME` | MongoDB database name                              | `puppet`                           |
| `DEBUG`       | Whether to enable debug logs                       | `false`                             |

A sample `.env` file might look like this:
```
SECRET_KEY=mySuperSecretKey
MODE=dev
MONGO_URI=mongodb://root:password@localhost:27017
MONGO_DB_NAME=puppet
DEBUG=true
```

For production, change these values according to your security and hosting requirements.

---

## Installation

There are two main approaches to installing Vergo Puppet:

1. **Local Installation (Development)**: Ideal for iterative development and debugging.  
2. **Docker Installation (Production)**: Best for deployment on servers.

### Local Installation (Development)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/vergo-puppet.git
   cd vergo-puppet
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Linux or Mac
   # or
   venv\Scripts\activate.bat  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install --no-cache-dir --upgrade pip
   pip install --no-cache-dir -r requirements-dev.txt
   pip install --no-cache-dir -r requirements-prod.txt
   ```

4. **Install Spacy model dependencies** (example: French model):
   ```bash
   python3 -m spacy download fr_core_news_md
   ```

5. **Configure your environment**:  
   - Copy `.env` or create your own `.env.local` file if you want a different configuration.  
   - Make sure to update the values according to your development setup (e.g., `MODE=dev`).

### Docker Installation (Production)

> **Note**: The provided Docker setup is a sample. Adapt it to your environment as needed.

1. **Ensure Docker and Docker Compose are installed**:
   - [Install Docker](https://docs.docker.com/engine/install/)  
   - [Install Docker Compose](https://docs.docker.com/compose/install/)

2. **Clone the repository**:
   ```bash
   git clone https://github.com/username/vergo-puppet.git
   cd vergo-puppet
   ```

3. **Create and configure `.env`**:
   - Update `MODE=prod`.
   - Ensure `MONGO_URI` is reachable from inside the container (e.g., a Docker service or external host).

4. **Build and run using docker-compose**:
   ```bash
   docker-compose build
   docker-compose up -d
   ```
   This will spin up a container named `vergo_puppet` and map port `3004` on your machine to port `8000` inside the container. Visit `http://localhost:3004/docs` to see the Swagger UI in production mode.

---

## Development Server

To run the application locally in development mode (with hot-reload):
```bash
uvicorn app.main:app --reload
```
- Access API docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

If you’re running via Docker Compose in **dev mode**:
```bash
docker-compose up --build
```
- Access API docs at: [http://localhost:3004/docs](http://localhost:3004/docs)

---

## Running Tests

Vergo Puppet uses **pytest** for testing:

1. **Install pytest** (if not already):
   ```bash
   pip install pytest
   ```

2. **Run all tests**:
   ```bash
   pytest
   ```

3. **Run a specific test**:
   ```bash
   pytest tests/test_search.py -m focus -s
   ```

4. **Run coverage tests**:
   ```bash
   pytest --cov
   ```
   This will generate a coverage report indicating which lines or functions have test coverage.

---

## APIs Overview

All endpoints are protected by a Bearer Token. You must include:
```
-H "Authorization: Bearer <YOUR_TOKEN>"
```
in each request. Tokens are generated/validated using the `SECRET_KEY` in your `.env` file.

Below are the primary endpoints, with example `cURL` commands:

### 1. Create a Model
**Endpoint**: `POST /create_model`  
Example:
```bash
curl -X POST http://localhost/api/create_model \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{
    "name": "model1",
    "neural_network_type": "SIAMESE",
    "dictionary": [["token1", "token2"], ["token3", "token4"]],
    "glossary": ["token1", "token2", "token3", "token4"]
  }'
```

### 2. Train a Model
**Endpoint**: `POST /train_model`  
Example:
```bash
curl -X POST http://localhost/api/train_model \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{
    "name": "model1",
    "neural_network_type": "SIAMESE",
    "training_data": [
      [["token1", "token2"], ["token3", "token4"], 0.5],
      [["token1", "token3"], ["token3", "token4"], 0.75]
    ]
  }'
```

### 3. Search with a Model
**Endpoint**: `POST /search`  
Example:
```bash
curl -X POST http://localhost/api/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{
    "name": "model1",
    "neural_network_type": "SIAMESE",
    "vector": ["token1", "token2"]
  }'
```

### 4. Update a Model
**Endpoint**: `PATCH /update_model`  
Example:
```bash
curl -X PATCH http://localhost/api/update_model \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{
    "name": "model1",
    "neural_network_type": "SIAMESE",
    "dictionary": [["token1", "token2"], ["token3", "token4"]],
    "glossary": ["token1", "token2", "token3", "token4"]
  }'
```

### 5. List All Models
**Endpoint**: `GET /models`  
Example:
```bash
curl -X GET http://localhost/api/models \
  -H "Authorization: Bearer <YOUR_TOKEN>"
```

### 6. Test a Model
**Endpoint**: `POST /test`
Example:
```bash
curl -X POST http://localhost/api/test \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{
    "name": "model1",
    "neural_network_type": "SIAMESE",
    "test_data": [
      [["token1", "token2"], ["token3", "token4"], 0.5],
      [["token1", "token3"], ["token3", "token4"], 0.75]
    ]
  }'
```

### 7. Embedding Endpoints

#### Create an Embedding Model
**Endpoint**: `POST /embedding/create`
```bash
curl -X POST http://localhost/api/embedding/create \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{
    "model_name": "puppet-o5",
    "vocab_path": "embedding_vocab.json",
    "trainset_path": "embedding_train.json"
  }'
```

#### Train an Embedding Model
**Endpoint**: `POST /embedding/train`
```bash
curl -X POST http://localhost/api/embedding/train \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{
    "model_name": "puppet-o5",
    "vocab_path": "embedding_vocab.json",
    "trainset_path": "embedding_train.json"
  }'
```

#### Encode a Sentence
**Endpoint**: `POST /embedding/encode`
```bash
curl -X POST http://localhost/api/embedding/encode \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{
    "model_name": "puppet-o5",
    "sentence": "My red bicycle"
  }'
```

#### Compute Similarity
**Endpoint**: `POST /embedding/similarity`
```bash
curl -X POST http://localhost/api/embedding/similarity \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{
    "model_name": "puppet-o5",
    "sentence1": "un vélo rouge",
    "sentence2": "un grand chapeau"
  }'
```

### 8. Thing Endpoints

#### Store a Thing
**Endpoint**: `POST /thing`
```bash
curl -X POST http://localhost/api/thing \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{
    "model_encode_name": "puppet-o5",
    "collection_name": "demo",
    "id": "thing_123",
    "data": {"label": "Red bike", "description": "Ultra light racing bike"}
  }'
```

#### List Things
**Endpoint**: `POST /thing/list`
```bash
curl -X POST http://localhost/api/thing/list \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{"collection_name": "demo"}'
```

#### Search Things
**Endpoint**: `POST /thing/search`
```bash
curl -X POST http://localhost/api/thing/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{
    "model_encode_name": "puppet-o5",
    "collection_name": "demo",
    "sentence": "red bike",
    "top_k": 5
  }'
```

---

## Model Details

### Puppet-o1 (Simple Model)
- **Type**: Feedforward Neural Network  
- **Use Case**: Regression tasks  
- **Example Input**:
  ```json
  {
    "type": 3,
    "surface": 70,
    "pieces": 3,
    "floor": 2
  }
```

#### Measure an Embedding Model
**Endpoint**: `POST /embedding/mesure`
```bash
curl -X POST http://localhost/api/embedding/mesure \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{
    "model_name": "puppet-o5",
    "test_path": "o5_test.json"
  }'
```

### Puppet-o2 (GRU Model)
- **Type**: Gated Recurrent Unit (GRU)  
- **Use Case**: Sequence classification  
- **Example Input**:
  ```json
  ["create", "user", "account"]
  ```

### Puppet-o3 (SIAMESE Model)
- **Type**: Siamese Neural Network  
- **Use Case**: Similarity tasks  
- **Example Input**:
  ```json
  [["token1", "token2"], ["token3", "token4"]]
  ```

### Puppet-o4 (LSTM Model)
- **Type**: Long Short-Term Memory (LSTM)
- **Use Case**: Time-series forecasting
- **Example Input**:
  ```json
  {
    "time": "2024-12-19T00:00:00Z",
    "temp": 5.0
  }
  ```

### Puppet-o5 (Embedding Model)
- **Type**: Universal Sentence Encoder (LSTM pooling)
- **Use Case**: Semantic search and "thing" indexing
- **Example Input**:
  ```json
  {
    "model_name": "puppet-o5",
    "sentence": "My red bicycle"
  }
  ```

---

## Project Structure

A simplified overview of the main files and directories:

```
vergo-puppet/
├─ app/
│  ├─ apis/                  # FastAPI route definitions
│  ├─ neural_network/        # NN definitions (nn_simple, nn_gru, nn_siamese, nn_lstm)
│  ├─ services/              # Logging, DB services, etc.
│  ├─ usecases/              # Contains use case logic for training, searching, etc.
│  ├─ main.py                # FastAPI main application
│  └─ common.py              # Common utilities
├─ training_data/            # Example training data (mounted volume in Docker)
├─ test_data/                # Example test data (mounted volume in Docker)
├─ requirements-dev.txt      # Dependencies for development
├─ requirements-prod.txt     # Dependencies for production
├─ Dockerfile                # Docker build instructions
├─ docker-compose.yml        # Docker Compose configuration
├─ .env                      # Environment variables
├─ README.md                 # Project documentation (this file)
└─ ...
```

You may add or remove files as your project evolves.

---

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**  
   - **Solution**: Ensure all dependencies are installed:
     ```bash
     pip install -r requirements-prod.txt
     ```

2. **API not starting**  
   - **Solution**: Verify the correct port (`8000` locally, `3004` via Docker) is available.

3. **Invalid token**  
   - **Solution**: Ensure `.env` file has the correct `SECRET_KEY` and that you’re passing a valid Bearer token in requests.

4. **Connection refused to MongoDB**  
   - **Solution**: Check your `MONGO_URI` and ensure MongoDB is running and accessible from the environment you are using.

---

## Contributing

We welcome contributions! To get started:

1. **Fork** the repository and create a new branch.  
2. **Make your changes** with clear commit messages.  
3. **Submit a pull request** to the main branch.

Please follow the existing code style and add/update tests for any changed functionality.

---

**Thank you for using Vergo Puppet!**  
If you have any questions or need further assistance, feel free to open an issue or reach out to the maintainers.