export async function fetchRecommendations(payload) {
  const res = await fetch("http://127.0.0.1:8000/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return res.json();
}

export async function logSelection(user_id, item_id, interaction = 1) {
  const res = await fetch("http://127.0.0.1:8000/log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id, item_id, interaction })
  });
  return res.json();
}