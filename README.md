# Vergo Puppet

AI service for model management, training, and predictions.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Development Server](#development-server)
5. [Running Tests](#running-tests)
6. [APIs Overview](#apis-overview)
7. [Model Details](#model-details)
    - [Puppet-o1 (Simple Model)](#puppet-o1-simple-model)
    - [Puppet-o2 (GRU Model)](#puppet-o2-gru-model)
    - [Puppet-o3 (SIAMESE Model)](#puppet-o3-siamese-model)
    - [Puppet-o4 (LSTM Model)](#puppet-o4-lstm-model)
8. [Troubleshooting](#troubleshooting)
9. [Changelog](#changelog)

---

## Introduction

Vergo Puppet is an AI-powered service designed for efficient model management, training, and inference tasks. It supports multiple neural network architectures to tackle a wide range of problems, from sequence classification to time-series predictions.

### Key Features
- Modular neural network architecture.
- RESTful APIs for model lifecycle management.
- Pre-built support for GRU, SIAMESE, LSTM, and SimpleNN models.

---

## Prerequisites

- **Python 3.9+**
- **pip** (latest version recommended)
- Docker and Docker Compose (optional for containerized setup)

---

## Installation

To set up the environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/vergo-puppet.git
   cd vergo-puppet
   ```

2. Install dependencies:
   ```bash
   pip install --no-cache-dir --upgrade pip && \
   pip install --no-cache-dir -r requirements-dev.txt && \
   pip install --no-cache-dir -r requirements-prod.txt
   ```

3. Install Spacy model dependencies:
   ```bash
   python3 -m spacy download fr_core_news_md
   ```

---

## Development Server

Run the development server locally with:
```bash
uvicorn app.main:app --reload
```

Access the API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Running Tests

Install `pytest`:
```bash
pip install pytest
```

Run all tests:
```bash
pytest
```

Run specific tests:
```bash
pytest tests/test_search.py -m focus -s
```

---

## APIs Overview

The Vergo Puppet service exposes a suite of RESTful APIs to interact with the models. Below are the primary endpoints, along with example `cURL` commands:

### Create a Model
**Endpoint**: `POST /create_model`

Example cURL:
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

---

### Train a Model
**Endpoint**: `POST /train_model`

Example cURL:
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

---

### Search with a Model
**Endpoint**: `POST /search`

Example cURL:
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

---

### Update a Model
**Endpoint**: `PATCH /update_model`

Example cURL:
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

---

### List All Models
**Endpoint**: `GET /models`

Example cURL:
```bash
curl -X GET http://localhost/api/models \
  -H "Authorization: Bearer <YOUR_TOKEN>"
```

---

### Test a Model
**Endpoint**: `POST /test`

Example cURL:
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

---

## Model Details

### Puppet-o1 (Simple Model)
- **Type**: Feedforward Neural Network.
- **Use Case**: Regression tasks.
- **Example Input**:
  ```json
  {
    "type": 3,
    "surface": 70,
    "pieces": 3,
    "floor": 2
  }
  ```

### Puppet-o2 (GRU Model)
- **Type**: Gated Recurrent Unit (GRU).
- **Use Case**: Sequence classification.
- **Example Input**:
  ```json
  ["create", "user", "account"]
  ```

### Puppet-o3 (SIAMESE Model)
- **Type**: Siamese Neural Network.
- **Use Case**: Similarity tasks.
- **Example Input**:
  ```json
  [["token1", "token2"], ["token3", "token4"]]
  ```

### Puppet-o4 (LSTM Model)
- **Type**: Long Short-Term Memory (LSTM).
- **Use Case**: Time-series forecasting.
- **Example Input**:
  ```json
  {
    "time": "2024-12-19T00:00:00Z",
    "temp": 5.0
  }
  ```

---

## Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError`
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements-prod.txt
```

#### Issue: API not starting
**Solution**: Verify the correct port (default: 8000) is available.

#### Issue: Invalid token
**Solution**: Ensure `.env` file has the correct `SECRET_KEY`.