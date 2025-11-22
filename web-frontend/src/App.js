import React, { useState } from "react";

function App() {
  const [items, setItems] = useState([]);
  const [userId, setUserId] = useState(1);
  const [recommendations, setRecommendations] = useState(null);

  const fetchItems = async () => {
    try {
      const res = await fetch("/api/items");
      const data = await res.json();
      setItems(data);
    } catch (err) {
      console.error("Error fetching items:", err);
    }
  };

  const addItem = async () => {
    try {
      const res = await fetch("/api/items", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: `Item ${items.length + 1}` }),
      });
      const data = await res.json();
      setItems([...items, data]);
    } catch (err) {
      console.error("Error adding item:", err);
    }
  };

  const createUser = async (payload) => {
    try {
      const res = await fetch("/api/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      return await res.json();
    } catch (err) {
      console.error("Error creating user:", err);
    }
  };

  const getRecommendations = async () => {
    try {
      const res = await fetch(`/api/recommendations/${userId}?top_k=5`);
      const data = await res.json();
      setRecommendations(data);
    } catch (err) {
      console.error("Error fetching recommendations:", err);
    }
  };

  const sendInteraction = async ({ user_id, nutrition_item_id = null, fitness_item_id = null, rating = 1 }) => {
    try {
      await fetch('/api/interactions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id, nutrition_item_id, fitness_item_id, rating })
      });
    } catch (err) {
      console.error('Error sending interaction', err);
    }
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h1>Recommender System Dashboard</h1>

      <div style={{ marginBottom: "1rem" }}>
        <button onClick={fetchItems}>Fetch Items</button>
        <button onClick={addItem} style={{ marginLeft: "1rem" }}>
          Add Item
        </button>
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <label>User ID: </label>
        <input type="number" value={userId} onChange={(e) => setUserId(Number(e.target.value))} style={{ width: 80 }} />
        <button onClick={getRecommendations} style={{ marginLeft: 8 }}>Get Recommendations</button>
      </div>

      <div style={{ display: "flex", gap: "2rem" }}>
        <div style={{ flex: 1 }}>
          <h2>Items</h2>
          <ul>
            {items.map((item, i) => (
              <li key={i}>
                <strong>{item.food || item.name}</strong>
                {item.calories !== undefined && (
                  <span> — {item.calories} kcal, {item.protein} g protein</span>
                )}
              </li>
            ))}
          </ul>
        </div>

        <div style={{ flex: 1 }}>
          <h2>Recommendations</h2>
          {!recommendations && <div>No recommendations yet.</div>}
          {recommendations && (
            <>
              <h3>Nutrition</h3>
              <ul>
                {recommendations.nutrition_items.map((it, i) => (
                  <li key={i}>
                    <strong>{it.food}</strong> — {it.calories} kcal, {it.protein} g protein
                    <button style={{ marginLeft: 8 }} onClick={() => sendInteraction({ user_id: userId, nutrition_item_id: it.id })}>Accept</button>
                  </li>
                ))}
              </ul>

              <h3>Fitness</h3>
              <ul>
                {recommendations.fitness_items.map((it, i) => (
                  <li key={i}>
                    <strong>{it.name}</strong> — {it.level}, {it.bodypart}
                    <button style={{ marginLeft: 8 }} onClick={() => sendInteraction({ user_id: userId, fitness_item_id: it.id })}>Accept</button>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
