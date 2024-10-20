# Vergo Puppet

AI for Vergo service

## Table of Contents

1. [Installation](#installation)
2. [Development Server](#development-server)
3. [Running Tests](#running-tests)
4. [APIs Overview](#apis-overview)
5. [Machine Learning](#machine-learning)
   - [SIAMESE Model](#siamese-model)

## Installation

To install dependencies, run:

```sh
pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt
```

## Development Server

To start the development server, use:

```sh
uvicorn app.main:app --reload
```

## Running Tests

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

## APIs Overview

The following APIs are available in the Vergo service, providing functionalities for model management, training, searching, and testing. The functionality depends on the model type (e.g., SIAMESE).

### ✉️ Model Management APIs

#### ✨ Creation of a Model

This endpoint allows the creation of a new model by specifying key parameters, such as the model name, neural network type, dictionary, and glossary. It facilitates the management of different models that can be used for training and searching.

<details>
<summary>📝 <code>POST</code> <code><b>/create_model</b></code> <code>(Create a new model)</code></summary>

#### Parameters

| Name              | Optional | Type          | Description                                                                                |
|-------------------|----------|---------------|--------------------------------------------------------------------------------------------|
| 🆔 name            | Required | string        | Name of the model to be created.                                                            |
| 🧠 neural_network_type | Required | string   | Type of neural network to create, such as `"SIAMESE"`.                                      |
| 📖 dictionary      | Required | list of lists | List of training pairs to use for building the model.                                       |
| 📚 glossary        | Required | list of strings | A list of all terms used across the dictionary entries.                                    |

#### Responses

| 📊 HTTP Code | 📄 Content Type       | 📝 Response                                                                 |
|--------------|----------------------|------------------------------------------------------------------------------|
| `201`        | `application/json`   | `{"message": "Model created successfully", "model_id": "model1"}`  |
| `400`        | `application/json`   | `{"error": "Missing or invalid parameter"}`                             |

##### 🛠️ Example CURL

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

</details>

#### ✨ Training a Model

This endpoint initiates training for a specified model using the provided training data.

<details>
<summary>📝 <code>POST</code> <code><b>/train_model</b></code> <code>(Train an existing model)</code></summary>

#### Parameters

| Name              | Optional | Type          | Description                                                                                |
|-------------------|----------|---------------|--------------------------------------------------------------------------------------------|
| 🆔 name            | Required | string        | Name of the model to be trained.                                                            |
| 🏋️‍♂️ training_data | Required | list of tuples | Training pairs consisting of input sequences and similarity scores.                          |

#### Responses

| 📊 HTTP Code | 📄 Content Type       | 📝 Response                                                                 |
|--------------|----------------------|------------------------------------------------------------------------------|
| `200`        | `application/json`   | `{"message": "Model trained successfully"}`                              |
| `400`        | `application/json`   | `{"error": "Invalid training data"}`                                     |

##### 🛠️ Example CURL

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

</details>

### 🔍 Searching and Testing APIs

#### ✨ Searching with a Model

This endpoint allows clients to search using the trained model to retrieve relevant results or similarity scores based on the input query.

<details>
<summary>📝 <code>POST</code> <code><b>/search</b></code> <code>(Search using a model)</code></summary>

#### Parameters

| Name              | Optional | Type          | Description                                                                                |
|-------------------|----------|---------------|--------------------------------------------------------------------------------------------|
| 🆔 name            | Required | string        | Name of the model to be used for the search.                                               |
| 🔍 vector          | Required | list of strings | Input vector used to search for similar results.                                          |

#### Responses

| 📊 HTTP Code | 📄 Content Type       | 📝 Response                                                                 |
|--------------|----------------------|------------------------------------------------------------------------------|
| `200`        | `application/json`   | `{"results": [...]}`                                                       |
| `400`        | `application/json`   | `{"error": "Invalid vector format"}`                                     |

##### 🛠️ Example CURL

```bash
curl -X POST http://localhost/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "name": "model1",
    "vector": ["sit", "up"]
  }'
```

</details>

#### ✨ Testing a Model

This endpoint is used to test a specified model using the provided test data and return evaluation metrics.

<details>
<summary>📝 <code>POST</code> <code><b>/test</b></code> <code>(Test a model)</code></summary>

#### Parameters

| Name              | Optional | Type          | Description                                                                                |
|-------------------|----------|---------------|--------------------------------------------------------------------------------------------|
| 🆔 name            | Required | string        | Name of the model to be tested.                                                             |
| 🔄 neural_network_type | Required | string   | Type of neural network (e.g., `"SIAMESE"`).                                                 |
| 🧪 test_data       | Required | list of tuples | Test data consisting of input pairs and expected similarity scores.                         |

#### Responses

| 📊 HTTP Code | 📄 Content Type       | 📝 Response                                                                 |
|--------------|----------------------|------------------------------------------------------------------------------|
| `200`        | `application/json`   | `{"evaluation": {"accuracy": 0.95}}`                                      |
| `400`        | `application/json`   | `{"error": "Invalid test data"}`                                         |

##### 🛠️ Example CURL

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

</details>

### ✨ Listing Models

Provides a list of all models that are currently available in the system.

<details>
<summary>📝 <code>GET</code> <code><b>/models</b></code> <code>(List available models)</code></summary>

#### Responses

| 📊 HTTP Code | 📄 Content Type       | 📝 Response                                                                 |
|--------------|----------------------|------------------------------------------------------------------------------|
| `200`        | `application/json`   | `{"models": ["model1", "model2"]}`                                      |

##### 🛠️ Example CURL

```bash
curl -X GET http://localhost/api/models
```

</details>

### ✨ API Version

Retrieves the current version of the API in use.

<details>
<summary>📝 <code>GET</code> <code><b>/version</b></code> <code>(Get API version)</code></summary>

#### Responses

| 📊 HTTP Code | 📄 Content Type       | 📝 Response                                                                 |
|--------------|----------------------|------------------------------------------------------------------------------|
| `200`        | `application/json`   | `{"version": "1.0.0"}`                                                  |

##### 🛠️ Example CURL

```bash
curl -X GET http://localhost/api/version
```

</details>

## Machine Learning

### SIAMESE Model

The SIAMESE model is a type of neural network architecture used primarily for tasks involving similarity, such as comparing two inputs to determine how similar they are. This model is particularly useful for problems such as facial recognition, signature verification, or any scenario where the goal is to identify how closely two inputs match.

The SIAMESE model helps in various applications by providing reliable similarity scoring, which can be leveraged for identity verification, product recommendations, and more. The model learns to generate embedding vectors for each input, which can then be compared using a similarity measure, such as cosine similarity.

#### APIs for SIAMESE Model

The following APIs are specifically used when working with the SIAMESE model:

- **/create_model**: Creates a new SIAMESE model. Requires parameters such as `name`, `dictionary`, `glossary`, and `neural_network_type` set to `"SIAMESE"`.
  - **Input Format**: JSON object containing `name` (str), `dictionary` (list of list of strings), `glossary` (list of strings), and `neural_network_type` (`"SIAMESE"`).
  - **Example**:
    ```json
    {
      "name": "model1",
      "neural_network_type": "SIAMESE",
      "dictionary": [["air", "squat"], ["push", "up"]],
      "glossary": ["air", "squat", "push", "up"]
    }
    ```

- **/train_model**: Initiates training for the SIAMESE model using pairs of data to learn similarity relationships.
  - **Input Format**: JSON object containing `name` (str) and `training_data` (list of tuples), where each tuple contains two lists of tokens (`siamese1`, `siamese2`) and optionally a similarity score (float).
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
  - **Input Format**: JSON object containing `name` (str) and `vector` (list of strings).
  - **Example**:
    ```json
    {
      "name": "model1",
      "vector": ["sit", "up"]
    }
    ```

- **/test**: Tests the SIAMESE model with specific pairs of inputs to evaluate the model's performance.
  - **Input Format**: JSON object containing `name` (str), `neural_network_type` (`"SIAMESE"`), and `test_data` (list of tuples), where each tuple contains two lists of tokens and a similarity score (float).
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

### SimpleNN Model

The `SimpleNN` model is used for regression tasks, such as predicting property prices based on input features like surface area, number of rooms, floor, and neighborhood. The following APIs allow you to create, train, search, and test a `SimpleNN` model.

#### APIs for SimpleNN Model

The following APIs are used to create, train, search, and test the `SimpleNN` model.

- **/create_model**: Creates a new `SimpleNN` model. Requires parameters such as `name` and `neural_network_type` set to `"SimpleNN"`.
  - **Input Format**: JSON object containing `name` (str) and `neural_network_type` (`"SimpleNN"`).
  - **Example**:
    ```json
    {
      "name": "puppet-o1",
      "neural_network_type": "SimpleNN"
    }
    ```

- **/train_model**: Initiates training for the `SimpleNN` model using structured training data (e.g., real estate data).
  - **Input Format**: JSON object containing `name` (str) and `training_data` (list of dictionaries with parameters like `type`, `surface`, `pieces`, `price`, etc.).
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
          "balcon": 0,
          "ascenseur": 1,
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
          "balcon": 1,
          "ascenseur": 1,
          "orientation": 0,
          "transports": 1,
          "neighborhood": 2,
          "price": 130000
        }
      ]
    }
    ```

- **/search**: Searches using the trained `SimpleNN` model to predict a result based on an input vector.
  - **Input Format**: JSON object containing `name` (str), `neural_network_type` (`"SimpleNN"`), and `vector` representing property characteristics (e.g., `type`, `surface`, `pieces`, etc.).
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
        "balcon": 0,
        "ascenseur": 0,
        "orientation": 3,
        "transports": 1,
        "neighborhood": 1
      }
    }
    ```

- **/test**: Tests a `SimpleNN` model using test data and returns evaluation metrics (e.g., prediction accuracy).
  - **Input Format**: JSON object containing `name` (str), `neural_network_type` (`"SimpleNN"`), and `test_data` (list of dictionaries similar to the training data).
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
          "balcon": 0,
          "ascenseur": 1,
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
          "balcon": 1,
          "ascenseur": 1,
          "orientation": 0,
          "transports": 1,
          "neighborhood": 2,
          "price": 130000
        }
      ]
    }
    ```