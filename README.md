# vergo_puppet
AI for Vergo service

## Dev

```
pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt
```

### Start

`uvicorn app.main:app --reload`

### Tests

`pip install pytest`
`pytest tests/`
`pytest tests/test_search.py -m focus -s` + `@pytest.mark.focus`