# Model Details

## Puppet-o1 (Simple Model)
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

### Measure an Embedding Model
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

## Puppet-o2 (GRU Model)
- **Type**: Gated Recurrent Unit (GRU)
- **Use Case**: Sequence classification
- **Example Input**:
```json
["create", "user", "account"]
```

## Puppet-o3 (SIAMESE Model)
- **Type**: Siamese Neural Network
- **Use Case**: Similarity tasks
- **Example Input**:
```json
[["token1", "token2"], ["token3", "token4"]]
```

## Puppet-o4 (LSTM Model)
- **Type**: Long Short-Term Memory (LSTM)
- **Use Case**: Time-series forecasting
- **Example Input**:
```json
{
  "time": "2024-12-19T00:00:00Z",
  "temp": 5.0
}
```

## Puppet-o5 (Embedding Model)
- **Type**: Universal Sentence Encoder (LSTM pooling)
- **Use Case**: Semantic search and "thing" indexing
- **Example Input**:
```json
{
  "model_name": "puppet-o5",
  "sentence": "My red bicycle"
}
```
