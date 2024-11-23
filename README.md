# Vergo Puppet

AI for the Vergo service

## Table of Contents

1. [Installation](#installation)
2. [Development Server](#development-server)
3. [Running Tests](#running-tests)
4. [APIs Overview](#apis-overview)
5. [Machine Learning](#machine-learning)
   - [Puppet-o1 (Simple Model)](#puppet-o1-simple-model)
   - [Puppet-o2 (GRU Model)](#puppet-o2-gru-model)
   - [Puppet-o3 (SIAMESE Model)](#puppet-o3-siamese-model)
   - [Puppet-o4 (LSTM Model)](#puppet-o4-lstm-model)

# Installation

To install dependencies, run:

```sh
pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt
```

Install spacy dependencies

```sh
python3 -m spacy download fr_core_news_md
```

# Development Server

To start the development server, use:

```sh
uvicorn app.main:app --reload
```

# Running Tests

To install pytest, run:

```sh
pip install pytest
```

To run all tests, use:

```sh
pytest
```

To run focused tests (marked with `@pytest.mark.focus`):

```sh
pytest tests/test_search.py -m focus -s
```

# APIs Overview

The following APIs are available in the Vergo service, providing functionalities for model management, training, searching, and testing. The functionality depends on the model type (e.g., SIAMESE).

## ✉️ Model Management APIs

### ✨ Creating a Model

This endpoint allows the creation of a new model by specifying key parameters, such as the model name, neural network type, dictionary, and glossary.

**Method**: `POST` `/create_model`

#### Parameters

- **name** *(string, required)*: Name of the model to create.
- **neural_network_type** *(string, required)*: Type of neural network to create, e.g., `"SIAMESE"`.
- **dictionary** *(list of lists, required)*: List of training pairs to use for building the model.
- **glossary** *(list of strings, required)*: A list of all terms used across the dictionary entries.

#### Responses

| HTTP Code | Content Type        | Response                                                        |
|-----------|---------------------|-----------------------------------------------------------------|
| `201`     | `application/json`  | `{"message": "Model created successfully", "model_id": "model1"}` |
| `400`     | `application/json`  | `{"error": "Missing or invalid parameter"}`                     |

**Example CURL**

```bash
curl -X POST http://localhost/api/create_model \
  -H "Content-Type: application/json" \
  -d '{
    "name": "model1",
    "neural_network_type": "SIAMESE",
    "dictionary": [["air", "squat"], ["push", "up"]],
    "glossary": ["air", "squat", "push", "up"]
  }'
```

### ✨ Training a Model

This endpoint initiates training for a specified model using the provided training data.

**Method**: `POST` `/train_model`

#### Parameters

- **name** *(string, required)*: Name of the model to be trained.
- **training_data** *(list of tuples, required)*: Training pairs consisting of input sequences and similarity scores.

#### Responses

| HTTP Code | Content Type        | Response                                      |
|-----------|---------------------|-----------------------------------------------|
| `200`     | `application/json`  | `{"message": "Model trained successfully"}`   |
| `400`     | `application/json`  | `{"error": "Invalid training data"}`          |

**Example CURL**

```bash
curl -X POST http://localhost/api/train_model \
  -H "Content-Type: application/json" \
  -d '{
    "name": "model1",
    "training_data": [
      [["dog", "cat", "bird"], ["dog", "cat", "bird"], 1.0],
      [["dog", "cat", "bird"], ["lion", "elephant", "bird"], 0.4]
    ]
  }'
```

## 🔍 Searching and Testing APIs

### ✨ Searching with a Model

This endpoint allows clients to search using the trained model to retrieve relevant results or similarity scores based on the input query.

**Method**: `POST` `/search`

#### Parameters

- **name** *(string, required)*: Name of the model to use for the search.
- **vector** *(list of strings, required)*: Input vector used to search for similar results.

#### Responses

| HTTP Code | Content Type        | Response                              |
|-----------|---------------------|---------------------------------------|
| `200`     | `application/json`  | `{"results": [...]}`                  |
| `400`     | `application/json`  | `{"error": "Invalid vector format"}`  |

**Example CURL**

```bash
curl -X POST http://localhost/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "name": "model1",
    "vector": ["sit", "up"]
  }'
```

### ✨ Testing a Model

This endpoint is used to test a specified model using the provided test data and return evaluation metrics.

**Method**: `POST` `/test`

#### Parameters

- **name** *(string, required)*: Name of the model to test.
- **neural_network_type** *(string, required)*: Type of neural network (e.g., `"SIAMESE"`).
- **test_data** *(list of tuples, required)*: Test data consisting of input pairs and expected similarity scores.

#### Responses

| HTTP Code | Content Type        | Response                                |
|-----------|---------------------|-----------------------------------------|
| `200`     | `application/json`  | `{"evaluation": {"accuracy": 0.95}}`    |
| `400`     | `application/json`  | `{"error": "Invalid test data"}`        |

**Example CURL**

```bash
curl -X POST http://localhost/api/test \
  -H "Content-Type: application/json" \
  -d '{
    "name": "model1",
    "neural_network_type": "SIAMESE",
    "test_data": [
      [["air", "squat"], ["air", "squat"], 1.0],
      [["push", "up"], ["pull", "up"], 0.5]
    ]
  }'
```

### ✨ Listing Models

Provides a list of all models that are currently available in the system.

**Method**: `GET` `/models`

#### Responses

| HTTP Code | Content Type        | Response                             |
|-----------|---------------------|--------------------------------------|
| `200`     | `application/json`  | `{"models": ["model1", "model2"]}`   |

**Example CURL**

```bash
curl -X GET http://localhost/api/models
```

### ✨ API Version

Retrieves the current version of the API in use.

**Method**: `GET` `/version`

#### Responses

| HTTP Code | Content Type        | Response                     |
|-----------|---------------------|------------------------------|
| `200`     | `application/json`  | `{"version": "1.0.0"}`       |

**Example CURL**

```bash
curl -X GET http://localhost/api/version
```

# Machine Learning

## Puppet-o1 (Simple Model)

The `SimpleNN` model is used for regression tasks, such as predicting property prices based on input features like surface area, number of rooms, floor, and neighborhood.

### APIs for Puppet-o1

- **/create_model**: Creates a new `SimpleNN` model.
  - **Example**:
    ```json
    {
      "name": "puppet-o1",
      "neural_network_type": "SimpleNN"
    }
    ```

- **/train_model**: Trains the `SimpleNN` model using structured training data.
  - **Example**:
    ```json
    {
      "name": "puppet-o1",
      "neural_network_type": "SimpleNN",
      "training_data": [
        {
          "type": 4,
          "surface": 98,
          "pieces": 4,
          "floor": 5,
          "parking": 1,
          "balcony": 0,
          "elevator": 1,
          "orientation": 6,
          "transports": 1,
          "neighborhood": 1,
          "price": 215000
        },
        {
          "type": 4,
          "surface": 68,
          "pieces": 4,
          "floor": 5,
          "parking": 1,
          "balcony": 1,
          "elevator": 1,
          "orientation": 0,
          "transports": 1,
          "neighborhood": 2,
          "price": 130000
        }
      ]
    }
    ```

- **/search**: Uses the trained `SimpleNN` model to predict a result based on an input vector.
  - **Example**:
    ```json
    {
      "name": "puppet-o1",
      "neural_network_type": "SimpleNN",
      "vector": {
        "type": 3,
        "surface": 70,
        "pieces": 3,
        "floor": 2,
        "parking": 0,
        "balcony": 0,
        "elevator": 0,
        "orientation": 3,
        "transports": 1,
        "neighborhood": 1
      }
    }
    ```

- **/test**: Tests the `SimpleNN` model using test data and returns evaluation metrics.
  - **Example**:
    ```json
    {
      "name": "puppet-o1",
      "neural_network_type": "SimpleNN",
      "test_data": [
        {
          "type": 4,
          "surface": 98,
          "pieces": 4,
          "floor": 5,
          "parking": 1,
          "balcony": 0,
          "elevator": 1,
          "orientation": 6,
          "transports": 1,
          "neighborhood": 1,
          "price": 215000
        },
        {
          "type": 4,
          "surface": 68,
          "pieces": 4,
          "floor": 5,
          "parking": 1,
          "balcony": 1,
          "elevator": 1,
          "orientation": 0,
          "transports": 1,
          "neighborhood": 2,
          "price": 130000
        }
      ]
    }
    ```

## Puppet-o2 (GRU Model)

The GRU (Gated Recurrent Unit) model is a type of recurrent neural network used primarily for sequence classification tasks, such as determining the category of a given sequence of tokens.

### APIs for Puppet-o2

- **/create_model**: Creates a new GRU model.

  - **Example**:

    ```json
    {
      "name": "puppet-o2",
      "neural_network_type": "GRU"
    }
    ```

- **/train_model**: Trains the GRU model using labeled training data to learn classification tasks.

  - **Example**:

    ```json
    {
      "name": "puppet-o2",
      "neural_network_type": "GRU",
      "training_data": [
        {
          "tokens": [
            "ne",
            "arriver",
            "connecter",
            "[service_k]",
            "me",
            "activer",
            "compte"
          ],
          "category": "probleme_connexion"
        },
        {
          "tokens": [
            "modification",
            "mot",
            "passe",
            "session",
            "[outlook]",
            "parvenir",
            "connecter",
            "[service_k]",
            "je",
            "avoir",
            "tester",
            "ancien",
            "mot",
            "passe",
            "fonctionner",
            "non",
            "[mail]",
            "[name]"
          ],
          "category": "probleme_connexion"
        }
      ]
    }
    ```

- **/search**: Uses the GRU model to predict the category of a given input sequence.

  - **Example**:

    ```json
    {
      "name": "puppet-o2",
      "neural_network_type": "GRU",
      "vector": [
        "ticket",
        "créer",
        "suite",
        "e-mail",
        "création",
        "utilisateur",
        "coffre",
        "banque",
        "sarl"
      ]
    }
    ```

- **/test**: Tests the GRU model with specific inputs to evaluate the model's performance.

  - **Example**:

    ```json
    {
      "name": "puppet-o2",
      "neural_network_type": "GRU",
      "test_data": [
        {
          "tokens": [
            "après-midi",
            "impossible",
            "accéder",
            "mot",
            "passer",
            "stocker",
            "[service_k]",
            "voir",
            "image",
            "joindre"
          ],
          "category": "gestion_compte"
        },
        {
          "tokens": [
            "ticket",
            "créer",
            "suite",
            "appel",
            "déblocage",
            "utilisateur",
            "[name]"
          ],
          "category": "gestion_acces"
        }
      ]
    }
    ```

- **/tokenize**: Tokenizes raw text data to prepare it for training or classification.

  - **Example**:

    ```json
    {
      "data": [
        {
          "description": "Bonjour, je n'arrive pas à me connecter à service, est-il possible de m'activer le compte. Merci"
        },
        {
          "description": "Bonjour, depuis la modification de mon mot de passe de session Outlook, je ne parviens plus à me connecter à [service_k]. J'ai testé avec l'ancien mot de passe mais cela ne fonctionne pas non plus!"
        }
      ]
    }
    ```

## Puppet-o3 (SIAMESE Model)

The SIAMESE model is a type of neural network architecture used primarily for tasks involving similarity, such as comparing two inputs to determine how similar they are.

### APIs for Puppet-o3

- **/create_model**: Creates a new SIAMESE model.
  - **Example**:
    ```json
    {
      "name": "model1",
      "neural_network_type": "SIAMESE",
      "dictionary": [["air", "squat"], ["push", "up"]],
      "glossary": ["air", "squat", "push", "up"]
    }
    ```

- **/train_model**: Trains the SIAMESE model using pairs of data to learn similarity relationships.
  - **Example**:
    ```json
    {
      "name": "model1",
      "training_data": [
        [["dog", "cat", "bird"], ["dog", "cat", "bird"], 1.0],
        [["dog", "cat", "bird"], ["lion", "elephant", "bird"], 0.4]
      ]
    }
    ```

- **/search**: Searches using the SIAMESE model to find the similarity between the given input vector and existing data.
  - **Example**:
    ```json
    {
      "name": "model1",
      "vector": ["sit", "up"]
    }
    ```

- **/test**: Tests the SIAMESE model with specific pairs of inputs to evaluate the model's performance.
  - **Example**:
    ```json
    {
      "name": "model1",
      "neural_network_type": "SIAMESE",
      "test_data": [
        [["air", "squat"], ["air", "squat"], 1.0],
        [["push", "up"], ["pull", "up"], 0.5]
      ]
    }
    ```

## Puppet-o4 (LSTM Model)

The LSTM (Long Short-Term Memory) model is a type of recurrent neural network used primarily for sequence prediction tasks, such as forecasting future values based on time-series data.

### APIs for Puppet-o4

- **/create_model**: Creates a new LSTM model.
  - **Example**:
    ```json
    {
      "name": "puppet-o4",
      "neural_network_type": "LSTM"
    }
    ```

- **/train_model**: Trains the LSTM model using historical time-series data.
  - **Example**:
    ```json
    {
      "name": "puppet-o4",
      "neural_network_type": "LSTM",
      "training_data": [
        {
          "time": "2018-01-01T00:00:00.000",
          "temp": 5.4,
          "dwpt": 5.0,
          "rhum": 97.0,
          "prcp": null,
          "snow": null,
          "wdir": 50.0,
          "wspd": 16.6,
          "wpgt": null,
          "pres": 1016.0,
          "tsun": null,
          "coco": null
        },
        {
          "time": "2018-01-01T01:00:00.000",
          "temp": 7.7,
          "dwpt": 5.5,
          "rhum": 86.0,
          "prcp": 2.0,
          "snow": null,
          "wdir": 150.0,
          "wspd": 11.2,
          "wpgt": null,
          "pres": 1018.3,
          "tsun": null,
          "coco": null
        }
      ]
    }
    ```

- **/search**: Uses the LSTM model to predict future values based on input features.
  - **Example**:
    ```json
    {
      "name": "puppet-o4",
      "neural_network_type": "LSTM",
      "vector": {
        "time": "2024-10-28T12:00:00.000",
        "dwpt": 10.0,
        "rhum": 70.0,
        "prcp": 0.0,
        "wdir": 180.0,
        "wspd": 5.0,
        "pres": 1015.0,
        "coco": 2
      }
    }
    ```

- **/test**: Tests the LSTM model with specific inputs to evaluate the model's performance.
  - **Example**:
    ```json
    {
      "name": "puppet-o4",
      "neural_network_type": "LSTM",
      "test_data": [
        {
          "time": "2021-02-24T00:00:00.000",
          "temp": 6.3,
          "dwpt": 5.4,
          "rhum": 94.0,
          "prcp": 0.0,
          "snow": null,
          "wdir": 0.0,
          "wspd": 0.0,
          "wpgt": null,
          "pres": 1037.6,
          "tsun": null,
          "coco": null
        },
        {
          "time": "2023-04-30T22:00:00.000",
          "temp": 12.8,
          "dwpt": 11.8,
          "rhum": 95.0,
          "prcp": 0.0,
          "snow": null,
          "wdir": 30.0,
          "wspd": 3.6,
          "wpgt": null,
          "pres": 1016.4,
          "tsun": null,
          "coco": 3.0
        }
      ]
    }
    ```

- **/create_data_puppet-o4**: Fetches and saves hourly weather data for Grenoble from 2018 to 2024.
    Data is collected from Meteostat and saved as a JSON file with key weather indicators.
  - **Example**:
    ```bash
    GET {{host}}/create_data_puppet-o4
    ```

# History

* 24/11/23 0.5.0 : Secure JWT