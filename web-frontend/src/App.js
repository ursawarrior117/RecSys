import React, { useState, useEffect } from "react";
import Recommendations from "./components/Recommendations";
import UserForm from "./components/UserForm";
import { logInteraction as apiLogInteraction } from "./services/api";
import { deleteUser as apiDeleteUser } from "./services/api";

function App() {
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
  const [selectedBodypart, setSelectedBodypart] = useState(null);
  const [bmr, setBmr] = useState(null);
  const [nutritionTargets, setNutritionTargets] = useState(null);
  const [mealPlan, setMealPlan] = useState(null);
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

  const getRecommendationsForUser = async (uid) => {
    try {
      const res = await fetch(`/api/recommendations/${uid}?top_k=5`);
      const data = await res.json();
      setRecommendations(data);
      // extract meal plan and nutrition stats if present
      if (data) {
        setTdee(data.tdee ?? null);
        setBmr(data.bmr ?? null);
        if (data.nutrition_targets) {
          setNutritionTargets(data.nutrition_targets);
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
            <div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: '1rem' }}>
                <button type="button" onClick={async () => {
                  try {
                    const res = await fetch(`/api/recommendations/${userId}?top_k=5&rec_type=all`);
                    const data = await res.json();
                    setRecommendations(data);
                    if (data) {
                      setTdee(data.tdee ?? null);
                      setBmr(data.bmr ?? null);
                      if (data.nutrition_targets) setNutritionTargets(data.nutrition_targets);
                      if (data.meal_plan) setMealPlan(data.meal_plan);
                    }
                  } catch (e) {
                    console.debug('Failed to fetch recommendations', e);
                  }
                }} style={buttonPrimaryStyle}>All Recommendations</button>
                
                <button type="button" onClick={async () => {
                  try {
                    const res = await fetch(`/api/recommendations/${userId}?top_k=5&rec_type=nutrition`);
                    const data = await res.json();
                    setRecommendations(data);
                    setSelectedBodypart(null);
                    if (data) {
                      setTdee(data.tdee ?? null);
                      setBmr(data.bmr ?? null);
                      if (data.nutrition_targets) setNutritionTargets(data.nutrition_targets);
                      if (data.meal_plan) setMealPlan(data.meal_plan);
                    }
                  } catch (e) {
                    console.debug('Failed to fetch recommendations', e);
                  }
                }} style={{...buttonPrimaryStyle, background: '#f59e0b'}}>Nutrition Only</button>
                
                <button type="button" onClick={async () => {
                  try {
                    const url = selectedBodypart 
                      ? `/api/recommendations/${userId}?top_k=5&rec_type=fitness&bodypart=${encodeURIComponent(selectedBodypart)}`
                      : `/api/recommendations/${userId}?top_k=5&rec_type=fitness`;
                    const res = await fetch(url);
                    const data = await res.json();
                    setRecommendations(data);
                    setMealPlan(null);
                    setNutritionTargets(null);
                    if (data) {
                      setTdee(data.tdee ?? null);
                      setBmr(data.bmr ?? null);
                    }
                  } catch (e) {
                    console.debug('Failed to fetch recommendations', e);
                  }
                }} style={{...buttonPrimaryStyle, background: '#10b981'}}>Fitness Only</button>
              </div>

              {/* Body Part Filter for Fitness */}
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginRight: 8 }}>Filter Fitness by Body Part:</label>
                <select 
                  value={selectedBodypart || ''} 
                  onChange={(e) => setSelectedBodypart(e.target.value || null)}
                  style={{
                    padding: '0.5rem 0.75rem',
                    borderRadius: 4,
                    border: '1px solid #d1d5db',
                    fontSize: 13,
                    background: '#fff',
                    cursor: 'pointer'
                  }}
                >
                  <option value="">All Body Parts</option>
                  <option value="chest">Chest</option>
                  <option value="back">Back</option>
                  <option value="legs">Legs</option>
                  <option value="shoulders">Shoulders</option>
                  <option value="arms">Arms</option>
                  <option value="core">Core</option>
                  <option value="glutes">Glutes</option>
                  <option value="cardio">Cardio</option>
                </select>
              </div>
            </div>
          )}
        </div>

        {/* User Stats Section */}
        {userCreated && tdee && (
          <div style={cardStyle}>
            <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: '1rem', color: '#1f2937' }}>Your Stats</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
              {bmr && (
                <div style={{ padding: '1rem', background: '#f0fdf4', borderRadius: 6, borderLeft: '4px solid #22c55e' }}>
                  <div style={{ fontSize: 12, color: '#666', marginBottom: 4 }}>Basal Metabolic Rate</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#16a34a' }}>{Math.round(bmr)}</div>
                  <div style={{ fontSize: 11, color: '#999' }}>kcal/day</div>
                </div>
              )}
              <div style={{ padding: '1rem', background: '#fef3c7', borderRadius: 6, borderLeft: '4px solid #f59e0b' }}>
                <div style={{ fontSize: 12, color: '#666', marginBottom: 4 }}>Daily Calorie Goal</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: '#d97706' }}>{Math.round(tdee)}</div>
                <div style={{ fontSize: 11, color: '#999' }}>kcal/day</div>
              </div>
              {nutritionTargets && nutritionTargets.protein_g && (
                <div style={{ padding: '1rem', background: '#dbeafe', borderRadius: 6, borderLeft: '4px solid #3b82f6' }}>
                  <div style={{ fontSize: 12, color: '#666', marginBottom: 4 }}>Daily Protein Goal</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#1d4ed8' }}>{Math.round(nutritionTargets.protein_g)}</div>
                  <div style={{ fontSize: 11, color: '#999' }}>grams/day</div>
                </div>
              )}
            </div>
          </div>
        )}

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
                                    <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
                                      {it.food || it.name}
                                      {it.reason === 'liked' && <span style={{ fontSize: 11, background: '#d1fae5', color: '#065f46', padding: '2px 6px', borderRadius: 3 }}>Liked</span>}
                                      {it.reason === 'snack' && <span style={{ fontSize: 11, background: '#fef3c7', color: '#92400e', padding: '2px 6px', borderRadius: 3 }}>Snack</span>}
                                    </div>
                                    <div style={{ fontSize: 12, color: '#666' }}>
                                      {it.category ? `${it.category}` : ''}
                                      {it.tags ? ` • ${it.tags}` : ''}
                                    </div>
                                  </div>
                                      <div style={{ textAlign: 'right', fontSize: 12, color: '#333' }}>
                                        <div>{Math.round(Number(it.calories || 0))} kcal</div>
                                        <div style={{ fontSize: 11, color: '#666' }}>{Math.round(Number(it.protein || 0))} g protein</div>
                                        {it.serving_multiplier && it.serving_multiplier > 1.05 && 
                                          <div style={{ fontSize: 11, color: '#666', fontStyle: 'italic' }}>x{it.serving_multiplier.toFixed(1)}</div>
                                        }
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

              {/* Fitness Recommendations Section */}
              {recommendations.fitness_items && recommendations.fitness_items.length > 0 && (
                <div style={{ marginTop: '2rem' }}>
                  <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: '1rem', color: '#374151' }}>Fitness Recommendations</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1rem' }}>
                    {recommendations.fitness_items.map((activity, idx) => (
                      <div
                        key={activity.id || idx}
                        style={{
                          border: '1px solid #e6e6e6',
                          borderRadius: 8,
                          padding: 16,
                          boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
                          background: '#fff',
                          transition: 'transform 0.2s, box-shadow 0.2s',
                          cursor: 'pointer'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.transform = 'translateY(-2px)';
                          e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.transform = 'translateY(0)';
                          e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.06)';
                        }}
                      >
                        <div style={{ marginBottom: 12 }}>
                          <div style={{ fontSize: 16, fontWeight: 700, color: '#1f2937', marginBottom: 4 }}>{activity.name}</div>
                          <div style={{ fontSize: 13, color: '#666', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                            <span style={{ background: '#f3f4f6', padding: '2px 8px', borderRadius: 12 }}>{activity.type}</span>
                            <span style={{ background: '#f3f4f6', padding: '2px 8px', borderRadius: 12 }}>{activity.level}</span>
                            <span style={{ background: '#f3f4f6', padding: '2px 8px', borderRadius: 12 }}>{activity.bodypart}</span>
                          </div>
                        </div>
                        
                        {activity.equipment && (
                          <div style={{ marginBottom: 10, fontSize: 13 }}>
                            <span style={{ color: '#666' }}>Equipment: </span>
                            <span style={{ fontWeight: 600 }}>{activity.equipment}</span>
                          </div>
                        )}

                        {activity.hybrid_score !== undefined && activity.hybrid_score !== null && (
                          <div style={{ marginBottom: 12, padding: 8, background: '#f0fdf4', borderRadius: 6, fontSize: 12 }}>
                            <div style={{ color: '#666' }}>Match Score</div>
                            <div style={{ fontSize: 18, fontWeight: 700, color: '#22c55e' }}>
                              {Math.round(activity.hybrid_score * 100)}%
                            </div>
                          </div>
                        )}

                        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
                          <button
                            onClick={async () => {
                              const fid = activity.id || null;
                              if (!fid) return;
                              if (!userCreated) {
                                alert('Create or load a user before interacting');
                                return;
                              }
                              setLikedItemIds(prev => {
                                const next = new Set(prev);
                                next.add(fid);
                                return next;
                              });
                              setDislikedItemIds(prev => {
                                const next = new Set(prev);
                                next.delete(fid);
                                return next;
                              });
                              try {
                                await apiLogInteraction({ user_id: userId, fitness_item_id: fid, event_type: 'like' });
                                await getRecommendationsForUser(userId);
                              } catch (e) {
                                console.error('Failed to log like interaction', e);
                                setLikedItemIds(prev => {
                                  const next = new Set(prev);
                                  next.delete(fid);
                                  return next;
                                });
                                alert('Failed to record like');
                              }
                            }}
                            disabled={likedItemIds.has(activity.id)}
                            style={{
                              flex: 1,
                              padding: '0.5rem',
                              borderRadius: 4,
                              background: likedItemIds.has(activity.id) ? '#d1fae5' : '#fff',
                              border: likedItemIds.has(activity.id) ? '1px solid #86efac' : '1px solid #d1d5db',
                              cursor: 'pointer',
                              fontSize: 13,
                              fontWeight: 600,
                              color: likedItemIds.has(activity.id) ? '#065f46' : '#333'
                            }}
                          >
                            {likedItemIds.has(activity.id) ? '✓ Liked' : 'Like'}
                          </button>

                          <button
                            onClick={async () => {
                              const fid = activity.id || null;
                              if (!fid) return;
                              if (!userCreated) {
                                alert('Create or load a user before interacting');
                                return;
                              }
                              setDislikedItemIds(prev => {
                                const next = new Set(prev);
                                next.add(fid);
                                return next;
                              });
                              setLikedItemIds(prev => {
                                const next = new Set(prev);
                                next.delete(fid);
                                return next;
                              });
                              try {
                                await apiLogInteraction({ user_id: userId, fitness_item_id: fid, event_type: 'dislike' });
                                await getRecommendationsForUser(userId);
                              } catch (e) {
                                console.error('Failed to log dislike interaction', e);
                                setDislikedItemIds(prev => {
                                  const next = new Set(prev);
                                  next.delete(fid);
                                  return next;
                                });
                                alert('Failed to record dislike');
                              }
                            }}
                            disabled={dislikedItemIds.has(activity.id)}
                            style={{
                              flex: 1,
                              padding: '0.5rem',
                              borderRadius: 4,
                              background: dislikedItemIds.has(activity.id) ? '#fee2e2' : '#fff',
                              border: dislikedItemIds.has(activity.id) ? '1px solid #fca5a5' : '1px solid #d1d5db',
                              cursor: 'pointer',
                              fontSize: 13,
                              fontWeight: 600,
                              color: dislikedItemIds.has(activity.id) ? '#7f1d1d' : '#333'
                            }}
                          >
                            {dislikedItemIds.has(activity.id) ? '✗ Disliked' : 'Dislike'}
                          </button>
                        </div>
                      </div>
                    ))}
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
