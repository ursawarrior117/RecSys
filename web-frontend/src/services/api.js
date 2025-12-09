// Use relative path by default so CRA proxy (development) forwards requests to backend
const API_BASE = process.env.REACT_APP_API_BASE || "";

export async function createUser(userData) {
  try {
    // Use the /api prefix so CRA dev-server proxies requests to the backend
    const res = await fetch(`${API_BASE}/api/users`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData)
    });
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (e) {
    console.error('createUser failed', e);
    throw e;
  }
}

export async function loadUser(userId) {
  try {
    const res = await fetch(`${API_BASE}/api/users/${userId}`);
    if (!res.ok) {
      if (res.status === 404) {
        throw new Error('User not found');
      }
      throw new Error(`HTTP ${res.status}`);
    }
    return await res.json();
  } catch (e) {
    console.error('loadUser failed', e);
    throw e;
  }
}

export async function getUsers() {
  try {
    const res = await fetch(`${API_BASE}/api/users`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    console.error('getUsers failed', e);
    throw e;
  }
}

export async function fetchRecommendations(payload) {
  try {
    const res = await fetch(`${API_BASE}/api/recommendations/${payload.user_id}?top_k=${payload.top_k || 5}`);
    return await res.json();
  } catch (e) {
    console.error('fetchRecommendations failed', e);
    throw e;
  }
}

export async function logInteraction({ user_id, nutrition_item_id = null, fitness_item_id = null, event_type = 'impression', rating = null }) {
  const body = { user_id, event_type };
  if (nutrition_item_id) body['nutrition_item_id'] = nutrition_item_id;
  if (fitness_item_id) body['fitness_item_id'] = fitness_item_id;
  if (rating !== null) body['rating'] = rating;
  try {
    const res = await fetch(`${API_BASE}/api/interactions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    // try parse JSON, but tolerate non-JSON responses
    try {
      return await res.json();
    } catch (e) {
      return { status: res.status, ok: res.ok };
    }
  } catch (e) {
    console.debug('logInteraction network error', e);
    return { status: 'network_error', error: String(e) };
  }
}

export async function deleteUser(userId) {
  try {
    const res = await fetch(`${API_BASE}/api/users/${userId}`, {
      method: 'DELETE'
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `HTTP ${res.status}`);
    }
    try {
      return await res.json();
    } catch (e) {
      return { status: res.status, ok: res.ok };
    }
  } catch (e) {
    console.error('deleteUser failed', e);
    throw e;
  }
}

export async function deleteAllUsers() {
  try {
    const res = await fetch(`${API_BASE}/api/users/clear`, {
      method: 'DELETE'
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `HTTP ${res.status}`);
    }
    try {
      return await res.json();
    } catch (e) {
      return { status: res.status, ok: res.ok };
    }
  } catch (e) {
    console.error('deleteAllUsers failed', e);
    throw e;
  }
}