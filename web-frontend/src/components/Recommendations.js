import React from "react";
import { logSelection } from "../services/api";

export default function Recommendations({ results, userId, onSelectCallback }) {
  if (!results) return null;
  const items = results.items || [];
  return (
    <div>
      <div>Targets — Calories: {results.targets?.tdee ?? "n/a"} | Protein: {results.targets?.protein_g ?? "n/a"}</div>
      <ul>
        {items.map(it => (
          <li key={it.item_id}>
            {it.food || it.name} - {it.calories ?? it.level ?? ""} 
            <button onClick={() => { logSelection(userId, it.item_id); if (onSelectCallback) onSelectCallback(it); }}>Select</button>
          </li>
        ))}
      </ul>
    </div>
  );
}