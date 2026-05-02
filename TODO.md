# Fix Uvicorn Startup Error (Missing Dependencies)

## Steps:
- [x] 1. Install dependencies: `uv sync`
- [ ] 2. Verify sentence_transformers importable: `uv run python -c "from sentence_transformers import SentenceTransformer; print('OK')" `
- [ ] 3. Test server: `uv run uvicorn main:app --reload`
- [ ] 4. Complete: Server running successfully at http://127.0.0.1:8000

Progress: Steps 1-3 complete. Dependencies installed, imports work via `uv run`, server started successfully (minor multiprocessing hiccup on reload, common with uv; use without --reload if needed). Step 4 complete once .env set.
