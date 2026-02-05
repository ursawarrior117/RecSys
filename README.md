# RecSys (Hybrid Recommender)

What it is
- A hybrid food + fitness recommender: content + collaborative signals.
- Backend: FastAPI (recsys_app.api). Models persisted to disk. DB via SQLAlchemy (DATABASE_URL).
- Frontend: React (web-frontend) that calls the API.

Prerequisites
- Python 3.10+ (3.11 is known to work; CPU TensorFlow recommended unless you have GPU), Node.js 16+ for frontend

Installation (Windows PowerShell)
1. Clone & enter repo
   git clone https://github.com/ursawarrior117/RecSys.git
   cd RecSys-main/RecSys-main

2. Backend (dev)
   # create and activate a virtual environment
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # upgrade pip and install dependencies
   python -m pip install --upgrade pip
   pip install -r requirements.txt

   # Optionally pin exact versions (reproducible installs)
   #   pip freeze > requirements.txt

3. Run API (dev)
   # run via module or use uvicorn for hot reload
   python main.py
   # or
   uvicorn recsys_app.api:app --host 127.0.0.1 --port 8000 --reload

4. Frontend
   cd web-frontend
   npm install
   npm start

5. Seed / Inspect DB
   # optional helpers (from repo root)
   python scripts/seed_db.py
   python scripts/inspect_db.py

Notes
- The repository excludes `.venv/` and large binary files from Git. Large artifacts were removed from history and a safety branch `backup-before-purge` was created (contains removed files). Use Git LFS or external storage for large assets.
- If you need GPU TensorFlow builds, install them separately following TensorFlow's official docs.

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