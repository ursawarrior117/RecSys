"""Small runner to start the FastAPI app created by recsys_app.api.

This file is intentionally lightweight — the application logic lives in
`recsys_app` and this module just exposes the `app` variable for Uvicorn
and provides a simple `__main__` entry so `python recommend_api.py` works.
"""

from recsys_app.api import create_app

# Create the FastAPI application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    # Run using the local app instance. Use reload in development.
    uvicorn.run("recommend_api:app", host="0.0.0.0", port=8000)

@app.get("/")
def root():
    return {"status": "ok", "message": "RecSys API running"}

def _background_retrain(tmp_users_df, n_df, f_df):
    """
    Background retrain task. Runs asynchronously via FastAPI BackgroundTasks.
    tmp_users_df: pandas.DataFrame of user feature rows
    n_df / f_df: nutrition and fitness item DataFrames
    """
    global nutrition_rec, fitness_rec
    try:
        # create simulated matrices (function exists in codebase)
        n_mat = simulate_interactions(tmp_users_df, n_df, item_type="nutrition")
        f_mat = simulate_interactions(tmp_users_df, f_df, item_type="fitness")

        try:
            nutrition_rec.train(tmp_users_df, n_df.reset_index(drop=True), np.array(n_mat))
            if save_obj:
                try:
                    save_obj(nutrition_rec, NUTR_MODEL_PATH)
                except Exception:
                    logging.exception("Failed to persist nutrition model")
        except Exception:
            logging.exception("Background nutrition retrain failed")

        try:
            fitness_rec.train(tmp_users_df, f_df.reset_index(drop=True), np.array(f_mat))
            if save_obj:
                try:
                    save_obj(fitness_rec, FIT_MODEL_PATH)
                except Exception:
                    logging.exception("Failed to persist fitness model")
        except Exception:
            logging.exception("Background fitness retrain failed")
    except Exception:
        logging.exception("Background retrain task failed")
