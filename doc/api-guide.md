# API Usage Guide

All endpoints require a Bearer token in the `Authorization` header.
The examples below use `http://localhost` as the base URL.

## 1. Create a Model
**Endpoint**: `POST /create_model`
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

## 2. Train a Model
**Endpoint**: `POST /train_model`
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

## 3. Search with a Model
**Endpoint**: `POST /search`
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

## 4. Update a Model
**Endpoint**: `PATCH /update_model`
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

## 5. List All Models
**Endpoint**: `GET /models`
```bash
curl -X GET http://localhost/api/models \
  -H "Authorization: Bearer <YOUR_TOKEN>"
```

## 6. Test a Model
**Endpoint**: `POST /test`
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

## 7. Embedding Endpoints
### Create an Embedding Model
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

### Train an Embedding Model
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

### Encode a Sentence
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

### Compute Similarity
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

## 8. Thing Endpoints
### Store a Thing
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

### List Things
**Endpoint**: `POST /thing/list`
```bash
curl -X POST http://localhost/api/thing/list \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -d '{"collection_name": "demo"}'
```

### Search Things
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
