import React, { useState } from "react";
import { fetchRecommendations } from "../services/api";

export default function UserForm({ onResults }) {
  const [form, setForm] = useState({
    user_id: "user_1", age: 24, weight: 70, height: 170, gender: "M",
    activity_level: "medium", health_goals: "MG", sleep_good: 1, item_type: "nutrition", top_k: 10
  });

  const handleChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const handleFetch = async () => {
    const payload = { user: form, item_type: form.item_type, top_k: parseInt(form.top_k, 10) };
    const data = await fetchRecommendations(payload);
    onResults(data, form.user_id);
  };

  return (
    <div>
      <input name="user_id" value={form.user_id} onChange={handleChange} />
      <input name="age" value={form.age} onChange={handleChange} />
      <input name="weight" value={form.weight} onChange={handleChange} />
      <input name="height" value={form.height} onChange={handleChange} />
      <select name="gender" value={form.gender} onChange={handleChange}><option>M</option><option>F</option></select>
      <select name="activity_level" value={form.activity_level} onChange={handleChange}>
        <option value="low">low</option><option value="medium">medium</option><option value="high">high</option>
      </select>
      <select name="health_goals" value={form.health_goals} onChange={handleChange}><option>MG</option><option>WL</option></select>
      <select name="item_type" value={form.item_type} onChange={handleChange}><option value="nutrition">Nutrition</option><option value="fitness">Fitness</option></select>
      <button onClick={handleFetch}>Fetch Items</button>
    </div>
  );
}