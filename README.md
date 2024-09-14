# vergo_puppet
AI for Vergo service

## Dev

```
pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r /app/requirements.txt
```

### Start

`uvicorn app.main:app --reload`

### Tests

`pip install pytest`
`pytest tests/`