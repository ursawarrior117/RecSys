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

  return (
    <div>
      <div>Targets — Calories: {results.nutrition_targets?.calories ?? results.targets?.tdee ?? "n/a"} | Protein: {results.nutrition_targets?.protein_g ?? results.targets?.protein_g ?? "n/a"}</div>
      <ul>
        {localItems.map(it => {
          const id = it.id || it.item_id;
          const isSelected = id && selectedIds.has(id);
          return (
            <li key={id || Math.random()}>
              {it.food || it.name} - {it.calories ?? it.level ?? ""}
              <button onClick={() => handleSelect(it)} disabled={isSelected} style={{ marginLeft: 8 }}>
                {isSelected ? 'Selected' : 'Select'}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}