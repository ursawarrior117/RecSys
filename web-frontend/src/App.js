import React, { useState, useEffect } from "react";
import Recommendations from "./components/Recommendations";
import UserForm from "./components/UserForm";
import { logInteraction as apiLogInteraction } from "./services/api";
import { deleteUser as apiDeleteUser } from "./services/api";

function App() {
  const [items, setItems] = useState([]);
  const [userId, setUserId] = useState(1);
  const [recommendations, setRecommendations] = useState(null);
  const [userForm, setUserForm] = useState({
    age: '',
    weight: '',
    height: '',
    gender: '',
    activity_level: '',
    health_goals: '',
    sleep_good: 1
  });
  const [userCreated, setUserCreated] = useState(false);
  const [tdee, setTdee] = useState(null);
  const [nutritionGoal, setNutritionGoal] = useState(null);
  const [bmr, setBmr] = useState(null);
  const [nutritionTargets, setNutritionTargets] = useState(null);
  const [mealPlan, setMealPlan] = useState(null);
  const [selectedItemIds, setSelectedItemIds] = useState(new Set());
  const [likedItemIds, setLikedItemIds] = useState(new Set());
  const [dislikedItemIds, setDislikedItemIds] = useState(new Set());
  const [loadUserId, setLoadUserId] = useState('');
  const [users, setUsers] = useState([]);
  const [userSearchQuery, setUserSearchQuery] = useState('');
  const [userPageIndex, setUserPageIndex] = useState(0);
  const usersPerPage = 10;

  // expose a small global flag so child components can tell whether a real
  // user has been created. This prevents accidental impression logging when
  // the UI defaults userId to 1 but no explicit user was created.
  useEffect(() => {
    try {
      window.__USER_CREATED__ = !!userCreated;
    } catch (e) {
      // ignore in non-browser environments
    }
  }, [userCreated]);

  const handleUserFormChange = (e) => {
    const { name, value } = e.target;
    setUserForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleUserFormSubmit = async (e) => {
    // keep default form submit behavior but do NOT auto-fetch recommendations;
    // creating a user should be an explicit action that does not immediately
    // generate recommendations unless the operator chooses to do so.
    e.preventDefault();
    const payload = {
      ...userForm,
      age: Number(userForm.age),
      weight: Number(userForm.weight),
      height: Number(userForm.height),
      sleep_good: Number(userForm.sleep_good)
    };
    const res = await createUser(payload);
    if (res && res.id) {
      setUserId(res.id);
      setUserCreated(true);
    }
  };

  const getRecommendationsForUser = async (uid) => {
    try {
      const res = await fetch(`/api/recommendations/${uid}?top_k=5`);
      const data = await res.json();
      setRecommendations(data);
      // set TDEE/BMR and nutrition targets if present
      if (data) {
        setTdee(data.tdee ?? null);
        setNutritionGoal(data.nutrition_goal ?? null);
        setBmr(data.bmr ?? null);
        // if nutrition_targets present, map to state
        if (data.nutrition_targets) {
          setNutritionTargets(data.nutrition_targets);
        } else if (data.calories || data.protein_g) {
          // older fields fallback
          setNutritionTargets({ calories: data.tdee, protein_g: data.nutrition_goal });
        }
        if (data.meal_plan) {
          setMealPlan(data.meal_plan);
        } else {
          setMealPlan(null);
        }
      }
    } catch (err) {
      console.error("Error fetching recommendations:", err);
    }
  };

  // NOTE: impression events are sent by the `Recommendations` component
  // to avoid double-posting we do not send impressions here.

  const fetchItems = async () => {
    try {
      const res = await fetch("/api/items");
      const data = await res.json();
      setItems(data);
    } catch (err) {
      console.error("Error fetching items:", err);
    }
  };

  const loadExistingUser = async () => {
    const id = Number(loadUserId);
    if (!id) {
      alert('Enter a valid user ID to load');
      return;
    }
    try {
        const res = await fetch(`/api/users/${id}`);
      if (!res.ok) {
        alert('User not found');
        return;
      }
      const data = await res.json();
      setUserId(data.id);
      setUserCreated(true);
      // fetch recommendations immediately for loaded user
      await getRecommendationsForUser(data.id);
    } catch (e) {
      console.error('Error loading user', e);
      alert('Failed to load user');
    }
  };

  const fetchUsers = async (offset = 0) => {
    try {
      // fetch users with interaction stats
        const res = await fetch(`/api/users/stats?skip=${offset}&limit=${usersPerPage}`);
      if (!res.ok) return;
      const data = await res.json();
      setUsers(data || []);
      setUserPageIndex(Math.floor(offset / usersPerPage));
    } catch (e) {
      console.debug('Failed to fetch users', e);
    }
  };

  // load users on mount so dropdown is populated
  useEffect(() => {
    fetchUsers(0);
  }, []);

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



  const cardStyle = {
    border: '1px solid #e0e0e0',
    borderRadius: 8,
    padding: '1.5rem',
    background: '#fff',
    boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
    marginBottom: '1.5rem'
  };

  const buttonStyle = {
    padding: '0.6rem 1rem',
    borderRadius: 4,
    border: '1px solid #ddd',
    background: '#fff',
    cursor: 'pointer',
    fontSize: 14,
    fontWeight: 500,
    transition: 'all 0.2s'
  };

  const buttonPrimaryStyle = {
    ...buttonStyle,
    background: '#007bff',
    color: '#fff',
    border: 'none'
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif", background: '#f9fafb', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        <h1 style={{ marginBottom: '0.5rem', fontSize: 28, fontWeight: 700, color: '#1f2937' }}>Nutrition Recommender System</h1>
        <p style={{ color: '#6b7280', marginBottom: '2rem', fontSize: 14 }}>Create users, load recommendations, and track interactions</p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>

          {/* Create User Section (moved to UserForm component) */}
          <div style={cardStyle}>
            <UserForm onResults={async (data, uid) => {
              if (uid) {
                setUserId(uid);
                setUserCreated(true);
                // if component provided recommendations data, set it; otherwise leave to Get Recommendations button
                if (data) {
                  setRecommendations(data);
                  setTdee(data.tdee ?? null);
                  setNutritionGoal(data.nutrition_goal ?? null);
                  setBmr(data.bmr ?? null);
                  if (data.nutrition_targets) setNutritionTargets(data.nutrition_targets);
                  if (data.meal_plan) setMealPlan(data.meal_plan);
                }
                // refresh user listing
                fetchUsers(0);
              }
            }} />
          </div>

          {/* Load User Section */}
          <div style={cardStyle}>
            <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: '1rem', color: '#1f2937' }}>Load Existing User</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              <input 
                type="text" 
                placeholder="Search by ID, age, weight..." 
                value={userSearchQuery} 
                onChange={(e) => { setUserSearchQuery(e.target.value); setUserPageIndex(0); }}
                style={{ padding: '0.6rem', borderRadius: 4, border: '1px solid #ddd', fontSize: 14 }}
              />
              <select 
                value={loadUserId} 
                onChange={(e) => setLoadUserId(e.target.value)}
                size={6}
                style={{ padding: '0.6rem', borderRadius: 4, border: '1px solid #ddd', fontSize: 13, fontFamily: 'monospace' }}
              >
                <option value="">Select user...</option>
                {users.filter(u => 
                  userSearchQuery === '' || String(u.id).includes(userSearchQuery) || String(u.age).includes(userSearchQuery) || String(u.weight).includes(userSearchQuery)
                ).map((u) => (
                  <option key={u.id} value={u.id}>
                    {`ID ${u.id} • age:${u.age} wt:${u.weight}kg • ${u.interaction_count || 0} interactions`}
                  </option>
                ))}
              </select>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button onClick={async () => { if (!loadUserId) { alert('Select a user'); return; } await loadExistingUser(); }} style={buttonPrimaryStyle} title="Load and fetch recommendations for selected user">Load User</button>
                <button onClick={() => fetchUsers(userPageIndex * usersPerPage)} style={buttonStyle}>Refresh</button>
                <button onClick={() => { if (userPageIndex > 0) fetchUsers((userPageIndex - 1) * usersPerPage); }} style={buttonStyle}>← Prev</button>
                <button onClick={() => { fetchUsers((userPageIndex + 1) * usersPerPage); }} style={buttonStyle}>Next →</button>
                <span style={{ marginLeft: 'auto', alignSelf: 'center', color: '#6b7280', fontSize: 12 }}>Page {userPageIndex + 1}</span>
              </div>
              {userCreated && (
                <div style={{ padding: '0.75rem', background: '#d1fae5', borderRadius: 4, fontSize: 13, color: '#065f46' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div>✓ Current User ID: <strong>{userId}</strong></div>
                    <button
                      onClick={async () => {
                        if (!window.confirm('Delete this user and their interactions? This cannot be undone.')) return;
                        try {
                          // Immediately clear client-side user state and recommendations
                          // to prevent impression logging during the deletion flow.
                          window.__USER_CREATED__ = false;
                          setUserCreated(false);
                          setUserId(0);
                          setRecommendations(null);
                          setMealPlan(null);

                          await apiDeleteUser(userId);
                          await fetchUsers(userPageIndex * usersPerPage);
                          alert('User deleted');
                        } catch (e) {
                          console.error('Failed to delete user', e);
                          alert('Failed to delete user');
                        }
                      }}
                      style={{ padding: '0.25rem 0.6rem', borderRadius: 4, background: '#ef4444', color: '#fff', border: 'none', cursor: 'pointer' }}
                    >
                      Clear User
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

        </div>

        {/* Recommendations Section */}
        <div style={cardStyle}>
          <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: '1rem', color: '#1f2937' }}>Get Recommendations</h2>
          {!userCreated ? (
            <div style={{ padding: '1rem', background: '#fef2f2', borderRadius: 4, color: '#7f1d1d', fontSize: 13 }}>
              Create a user first using the form above.
            </div>
          ) : (
            <button type="button" onClick={async () => {
              try {
                await getRecommendationsForUser(userId);
              } catch (e) {
                console.debug('Failed to fetch recommendations', e);
              }
            }} style={buttonPrimaryStyle}>Get Recommendations</button>
          )}
        </div>

        {/* Recommendations Display */}
        {recommendations ? (
          <div style={cardStyle}>
              <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: '1.5rem', color: '#1f2937' }}>Recommendations & Meal Plan</h2>
              {mealPlan && (
                <div style={{ marginBottom: '2rem' }}>
                  <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: '0.8rem', color: '#374151' }}>Daily Meal Plan</h3>
                      {recommendations.daily_protein !== undefined && recommendations.daily_calories !== undefined && (
                        <div style={{ marginBottom: 8, fontSize: 13 }}>
                          <strong>Progress:</strong> {Math.round(recommendations.daily_protein || 0)} g protein • {Math.round(recommendations.daily_calories || 0)} kcal
                          {recommendations.percent_of_goal ? ` — ${Math.round(recommendations.percent_of_goal)}% of protein goal` : ''}
                        </div>
                      )}
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                    {Object.keys(mealPlan).map((mealName) => {
                      const items = mealPlan[mealName] || [];
                      // compute meal totals
                      const totals = items.reduce(
                        (acc, it) => {
                          acc.calories += Number(it.calories || 0);
                          acc.protein += Number(it.protein || 0);
                          acc.carbs += Number(it.carbohydrates || it.carbs_g || 0);
                          acc.fat += Number(it.fat || 0);
                          return acc;
                        },
                        { calories: 0, protein: 0, carbs: 0, fat: 0 }
                      );

                      return (
                        <div
                          key={mealName}
                          style={{
                            minWidth: 240,
                            flex: '1 1 240px',
                            border: '1px solid #e6e6e6',
                            borderRadius: 8,
                            padding: 12,
                            boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
                            background: '#fff'
                          }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                            <strong>{mealName}</strong>
                            <small style={{ color: '#666' }}>{items.length} items</small>
                          </div>

                          <ul style={{ padding: 0, margin: 0, listStyle: 'none' }}>
                            {items.map((it, i) => (
                              <li key={it.id || `${mealName}-${i}`} style={{ marginBottom: 8, paddingBottom: 8, borderBottom: '1px dashed #f0f0f0' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                                  <div>
                                    <div style={{ fontWeight: 600 }}>{it.food || it.name}</div>
                                    <div style={{ fontSize: 12, color: '#666' }}>
                                      {it.category ? `${it.category}` : ''}
                                      {it.tags ? ` • ${it.tags}` : ''}
                                    </div>
                                  </div>
                                      <div style={{ textAlign: 'right', fontSize: 12, color: '#333' }}>
                                        <div>{Math.round(Number(it.calories || 0))} kcal</div>
                                        <div style={{ fontSize: 11, color: '#666' }}>{Math.round(Number(it.protein || 0))} g protein</div>
                                      </div>
                                </div>
                                <div style={{ marginTop: 6, fontSize: 12, color: '#444' }}>
                                  <span style={{ marginRight: 10 }}>Carbs: {Math.round(Number(it.carbohydrates || it.carbs_g || 0))} g</span>
                                  <span>Fat: {Math.round(Number(it.fat || 0))} g</span>
                                </div>
                                    <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
                                      <button
                                        onClick={async () => {
                                          const nid = it.id || it.item_id || null;
                                          if (!nid) return;
                                          if (!userCreated) {
                                            alert('Create or load a user before interacting');
                                            return;
                                          }
                                          // optimistic UI: mark liked
                                          setLikedItemIds(prev => {
                                            const next = new Set(prev);
                                            next.add(nid);
                                            return next;
                                          });
                                          // ensure disliked state is cleared locally
                                          setDislikedItemIds(prev => {
                                            const next = new Set(prev);
                                            next.delete(nid);
                                            return next;
                                          });
                                          try {
                                            await apiLogInteraction({ user_id: userId, nutrition_item_id: nid, event_type: 'like' });
                                            // refresh recommendations to reflect CF signal
                                            await getRecommendationsForUser(userId);
                                          } catch (e) {
                                            console.error('Failed to log like interaction', e);
                                            setLikedItemIds(prev => {
                                              const next = new Set(prev);
                                              next.delete(nid);
                                              return next;
                                            });
                                            alert('Failed to record like');
                                          }
                                        }}
                                        disabled={likedItemIds.has(it.id || it.item_id)}
                                        style={{ marginTop: 6, padding: '0.3rem 0.6rem', borderRadius: 4, background: likedItemIds.has(it.id || it.item_id) ? '#d1fae5' : undefined }}
                                      >{likedItemIds.has(it.id || it.item_id) ? 'Liked' : 'Like'}</button>

                                      <button
                                        onClick={async () => {
                                          const nid = it.id || it.item_id || null;
                                          if (!nid) return;
                                          if (!userCreated) {
                                            alert('Create or load a user before interacting');
                                            return;
                                          }
                                          // optimistic UI: mark disliked
                                          setDislikedItemIds(prev => {
                                            const next = new Set(prev);
                                            next.add(nid);
                                            return next;
                                          });
                                          // ensure liked state is cleared locally
                                          setLikedItemIds(prev => {
                                            const next = new Set(prev);
                                            next.delete(nid);
                                            return next;
                                          });
                                          try {
                                            await apiLogInteraction({ user_id: userId, nutrition_item_id: nid, event_type: 'dislike' });
                                            // refresh recommendations to reflect CF signal
                                            await getRecommendationsForUser(userId);
                                          } catch (e) {
                                            console.error('Failed to log dislike interaction', e);
                                            setDislikedItemIds(prev => {
                                              const next = new Set(prev);
                                              next.delete(nid);
                                              return next;
                                            });
                                            alert('Failed to record dislike');
                                          }
                                        }}
                                        disabled={dislikedItemIds.has(it.id || it.item_id)}
                                        style={{ marginTop: 6, padding: '0.3rem 0.6rem', borderRadius: 4, background: dislikedItemIds.has(it.id || it.item_id) ? '#fee2e2' : undefined }}
                                      >{dislikedItemIds.has(it.id || it.item_id) ? 'Disliked' : 'Dislike'}</button>
                                    </div>
                              </li>
                            ))}
                          </ul>

                          <div style={{ marginTop: 10, borderTop: '1px solid #fafafa', paddingTop: 8, display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                            <div style={{ fontWeight: 600 }}>Meal total</div>
                            <div style={{ textAlign: 'right' }}>
                              <div>{Math.round(totals.calories)} kcal</div>
                              <div style={{ color: '#666' }}>{Math.round(totals.protein)} g protein</div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: '1rem', marginTop: '2rem', color: '#374151' }}>Recommended Items</h3>
              <Recommendations
                results={recommendations}
                userId={userId}
                onSelectCallback={async (item) => {
                  try {
                    await getRecommendationsForUser(userId);
                  } catch (e) {
                    console.debug('Failed to refresh recommendations', e);
                  }
                }}
              />
          </div>
        ) : (
          <div style={cardStyle}>
            <p style={{ color: '#6b7280', fontSize: 14 }}>Load a user and click "Get Recommendations" to see meal plans and recommendations.</p>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
