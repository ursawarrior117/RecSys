# RecSys (Hybrid Recommender)

What it is
- A hybrid food + fitness recommender: content + collaborative signals.
- Backend: FastAPI (recsys_app.api). Models persisted to disk. DB via SQLAlchemy (DATABASE_URL).
- Frontend: React (web-frontend) that calls the API.

Quick start (Windows PowerShell)
1. Backend (dev)
   # optional: set virtualenv or activate conda
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt

   # start API (recommended)
   python recommend_api.py
   # or
   uvicorn recommend_api:app --host 127.0.0.1 --port 8000 --reload

2. Frontend
   cd web-frontend
   npm install
   npm start

3. Seed DB (optional)
   python seed_db.py

4. Inspect DB
   python inspect_db.py

API
- POST /recommend  -> body: { user: { ... }, item_type: "nutrition"|"fitness", top_k: 10 }
- POST /log        -> body: { user_id, item_id, interaction }

Config (env)
- DATABASE_URL (default: sqlite:///recsys_users.db)
- MODEL_DIR (default: ./models)
- PORT / HOST (backend run options)
Use .env or environment variables.

Docker (quick)
- Build & run:
  docker-compose build
  docker-compose up

Notes
- Models are trained at startup (warm-start) and retrained in background when new interactions arrive.
- Use `inspect_db.py` to verify users/interactions.
- To move the DB: set DATABASE_URL to a Postgres/MySQL URI.

Next tidy tasks (recommended)
- Move remaining top-level helpers into recsys_app.
- Add Alembic migrations for users/interactions.
- Add unit tests and CI pipeline.