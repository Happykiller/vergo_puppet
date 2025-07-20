# Troubleshooting

## Common Issues

1. **ModuleNotFoundError**
   - **Solution**: Ensure all dependencies are installed:
     ```bash
     pip install -r requirements-prod.txt
     ```

2. **API not starting**
   - **Solution**: Verify the correct port (`8000` locally, `3004` via Docker) is available.

3. **Invalid token**
   - **Solution**: Ensure `.env` has the correct `SECRET_KEY` and include a valid Bearer token in requests.

4. **Connection refused to MongoDB**
   - **Solution**: Check your `MONGO_URI` and confirm MongoDB is running and reachable.
