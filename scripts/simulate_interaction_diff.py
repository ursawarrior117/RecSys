import os
import sys
import json
import asyncio
from pprint import pprint

# ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ['RECSYS_DISABLE_ML'] = '1'

from recsys_app.database.session import init_sample_data, SessionLocal
from recsys_app.routes.recommendations import get_recommendations
from recsys_app.routes.interactions import log_interaction


def to_simple_list(resp):
    # resp is Pydantic model RecommendationResponse
    try:
        d = json.loads(resp.json())
    except Exception:
        try:
            d = resp.model_dump() if hasattr(resp, 'model_dump') else dict(resp)
        except Exception:
            d = resp
    nut_ids = [it.get('id') for it in (d.get('nutrition_items') or [])]
    fit_ids = [it.get('id') for it in (d.get('fitness_items') or [])]
    return d, nut_ids, fit_ids


def main():
    init_sample_data()
    db = SessionLocal()
    try:
        user_id = 1
        print('Fetching recommendations BEFORE interaction for user', user_id)
        before = asyncio.run(get_recommendations(user_id=user_id, db=db, top_k=5))
        b_json, b_nut, b_fit = to_simple_list(before)

        print('\nTop nutrition IDs before:', b_nut)

        # choose an item that's NOT the top one to see movement; pick last of the returned list
        target_id = b_nut[-1] if b_nut else None
        print('Logging ACCEPT interaction for nutrition_item_id=', target_id)
        if target_id is None:
            print('No nutrition items available; aborting')
            return
        payload = {'user_id': user_id, 'nutrition_item_id': target_id, 'event_type': 'accept'}
        res = log_interaction(payload, db)
        print('log_interaction ->', res)

        print('\nFetching recommendations AFTER interaction for user', user_id)
        after = asyncio.run(get_recommendations(user_id=user_id, db=db, top_k=5))
        a_json, a_nut, a_fit = to_simple_list(after)

        print('\nTop nutrition IDs after:', a_nut)

        print('\nFull BEFORE JSON:')
        pprint(b_json)
        print('\nFull AFTER JSON:')
        pprint(a_json)

        # simple movement summary
        moved_up = [i for i in a_nut if i not in b_nut or a_nut.index(i) < b_nut.index(i) if i in b_nut]
        print('\nSummary: moved_up (ids):', moved_up)
    finally:
        db.close()

if __name__ == '__main__':
    main()
