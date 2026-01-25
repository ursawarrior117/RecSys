import os
import sys
# ensure repo root is on sys.path so `recsys_app` imports succeed when running scripts
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ['RECSYS_DISABLE_ML'] = '1'

from recsys_app.database.session import init_sample_data, SessionLocal
from recsys_app.routes.interactions import log_interaction
import asyncio

# initialize DB and sample data
init_sample_data()

db = SessionLocal()
try:
    payload = {'user_id': 1, 'nutrition_item_id': 1, 'event_type': 'accept'}
    print('Logging interaction:', payload)
    res = log_interaction(payload, db)
    print('log_interaction ->', res)

    # call recommendations route directly (it's async)
    from recsys_app.routes.recommendations import get_recommendations

    print('Requesting recommendations for user 1...')
    recs = asyncio.run(get_recommendations(user_id=1, db=db, top_k=5))
    try:
        # Pydantic model -> dict
        print('Recommendations response:')
        print(recs.json())
    except Exception:
        print(recs)
finally:
    db.close()
