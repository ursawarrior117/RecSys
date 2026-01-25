import React, { useEffect, useState, useMemo } from "react";
import { logInteraction } from "../services/api";

export default function Recommendations({ results, userId, onSelectCallback }) {
  // Wrap incoming in useMemo to stabilize the reference and prevent
  // unnecessary re-renders of useEffect dependencies
  const incoming = useMemo(
    () => (results && (results.nutrition_items || results.items)) || [],
    [results]
  );

  // local list state so we can optimistic-reorder without waiting for server
  const [localItems, setLocalItems] = useState(incoming || []);
  const [selectedIds, setSelectedIds] = useState(new Set());

  // keep localItems in sync when incoming results change
  useEffect(() => {
    setLocalItems(incoming || []);
  }, [incoming]);

  // send impressions for visible items once on mount/update (fire-and-forget)
  useEffect(() => {
    // Only send impressions when there is a known created user. The app
    // previously sent impressions when `userId` defaulted to 1 even if the
    // user wasn't explicitly created, polluting interaction logs. Require
    // `userCreated` to be true (passed from parent) to send impressions.
    if (!userId || !window.__USER_CREATED__) return;
    (async () => {
      try {
        const ids = (incoming || []).map(it => it.id || it.item_id).filter(Boolean);
        if (ids.length === 0) return;
        // send a single example impression for now (backend supports single-item calls)
        await logInteraction({ user_id: userId, nutrition_item_id: ids[0], event_type: 'impression' });
      } catch (e) {
        console.debug('impression logging failed', e);
      }
    })();
  }, [incoming, userId]);

  const handleSelect = async (it) => {
    const nid = it.id || it.item_id;
    if (!userId || !nid) return;
    // optimistic UI update: mark selected immediately and move item to top
    setSelectedIds(prev => new Set(prev).add(nid));
    setLocalItems(prev => {
      try {
        const without = (prev || []).filter(x => (x.id || x.item_id) !== nid);
        const picked = (prev || []).find(x => (x.id || x.item_id) === nid) || it;
        return [picked, ...without];
      } catch (e) {
        return prev;
      }
    });

    try {
      await logInteraction({ user_id: userId, nutrition_item_id: nid, event_type: 'accept' });
    } catch (e) {
      console.debug('accept logging failed', e);
    }
    if (onSelectCallback) onSelectCallback(it);
  };

  // Render a compact list of recommendations showing key nutrients
  if (!incoming || incoming.length === 0) return null;

  return (
    <div className="recommendations-list">
      {(localItems || incoming).map((it) => {
        const id = it.id || it.item_id;
        const food = it.food || it.name || '';
        const kcal = Number(it.calories || it.calories_val || 0).toFixed(0);
        const protein = Number(it.protein || it.protein_val || 0).toFixed(0);
        const carbs = Number(it.carbohydrates || it.carbs || 0).toFixed(0);
        const fat = Number(it.fat || 0).toFixed(0);
        const fiber = Number(it.fiber || it.fiber_val || 0).toFixed(0);
        const magnesium = Number(it.magnesium || it.magnesium_val || 0).toFixed(0);

        return (
          <div key={id} className="rec-item" style={{borderBottom: '1px solid #eee', padding: '12px 0'}}>
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
              <div style={{fontWeight: 600}}>{food}</div>
              <div style={{textAlign: 'right', color: '#666'}}>
                <div>{kcal} kcal</div>
                <div style={{fontSize: '0.9em'}}>{protein} g protein</div>
              </div>
            </div>
            <div style={{marginTop: 8, color: '#444', fontSize: '0.9em'}}>
              <span style={{marginRight: 12}}>Carbs: {carbs} g</span>
              <span style={{marginRight: 12}}>Fat: {fat} g</span>
              <span style={{marginRight: 12}}>Fiber: {fiber} g</span>
              <span>Magnesium: {magnesium} mg</span>
            </div>
            <div style={{marginTop: 8}}>
              <button onClick={() => handleSelect(it)} style={{marginRight: 8}}>Like</button>
              <button onClick={() => {}}>Dislike</button>
            </div>
          </div>
        );
      })}
    </div>
  );
}