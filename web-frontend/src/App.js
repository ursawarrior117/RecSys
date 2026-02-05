import React, { useState, useEffect } from "react";
// Recommendations component removed from main view; kept as a component for reuse if needed
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
  const [bmi, setBmi] = useState(null);
  const [userHeight, setUserHeight] = useState(null);
  const [userWeight, setUserWeight] = useState(null);
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
      console.log('[getRecommendationsForUser] API Response:', data);
      console.log('[getRecommendationsForUser] BMI:', data.bmi, 'Height:', data.user_height, 'Weight:', data.user_weight);
      setRecommendations(data);
      // extract meal plan and nutrition stats if present
      if (data) {
        setTdee(data.tdee ?? null);
        setBmi(data.bmi ?? null);
        setUserHeight(data.user_height ?? null);
        setUserWeight(data.user_weight ?? null);
        setBmr(data.bmr ?? null);
        console.log('[setters] After setState - bmi should be:', data.bmi, 'userHeight should be:', data.user_height);
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
    border: '1px solid #e2e8f0',
    borderRadius: '12px',
    padding: '24px',
    background: '#fff',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    marginBottom: '24px'
  };

  const buttonStyle = {
    padding: '10px 16px',
    borderRadius: '6px',
    border: '1px solid #cbd5e0',
    background: '#fff',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
    transition: 'all 0.2s ease',
    color: '#4a5568'
  };

  const buttonPrimaryStyle = {
    ...buttonStyle,
    background: '#3182ce',
    color: '#fff',
    border: 'none'
  };

  // Helper: simple SVG pie chart for TDEE distribution (BMR, Activity, TEF)
  function MealPieChart({ bmr, tdee, size = 140 }) {
    const tef = tdee ? Math.max(0, tdee * 0.1) : 0; // thermal effect ~10%
    // activity contribution = remaining calories after BMR and TEF
    let activity = 0;
    if (tdee && bmr) {
      activity = Math.max(0, tdee - bmr - tef);
    }
    const bmrVal = bmr ? Math.min(bmr, tdee || bmr) : 0;
    const parts = [
      { label: 'BMR', value: bmrVal, color: '#60a5fa' },
      { label: 'Activity', value: activity, color: '#34d399' },
      { label: 'Thermic Effect', value: tef, color: '#f59e0b' }
    ];
    const sum = parts.reduce((s, p) => s + p.value, 0) || (tdee || 1);

    // build segments
    let angle = -90;
    const segments = parts.map((p) => {
      const sweep = (p.value / sum) * 360;
      const start = angle;
      const end = angle + sweep;
      angle = end;
      return { start, end, color: p.color, label: p.label, value: p.value };
    });

    const arcPath = (cx, cy, r, startDeg, endDeg) => {
      const toRad = (d) => (d * Math.PI) / 180;
      const x1 = cx + r * Math.cos(toRad(startDeg));
      const y1 = cy + r * Math.sin(toRad(startDeg));
      const x2 = cx + r * Math.cos(toRad(endDeg));
      const y2 = cy + r * Math.sin(toRad(endDeg));
      const largeArc = endDeg - startDeg <= 180 ? '0' : '1';
      return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2} Z`;
    };

    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          {segments.map((s, i) => (
            <path key={i} d={arcPath(size / 2, size / 2, size / 2 - 6, s.start, s.end)} fill={s.color} stroke="#fff" strokeWidth={1} />
          ))}
          <circle cx={size / 2} cy={size / 2} r={size / 2 - 34} fill="#fff" />
          <text x="50%" y="50%" dominantBaseline="middle" textAnchor="middle" style={{ fontSize: 12, fontWeight: 700, fill: '#111827' }}>{Math.round(tdee || sum)} kcal</text>
        </svg>
        <div style={{ marginTop: 8, fontSize: 12, color: '#6b7280' }}>
          {segments.map((s, i) => (
            <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <span style={{ width: 10, height: 10, background: s.color, display: 'inline-block', borderRadius: 2 }} />
              <span style={{ color: '#374151', fontSize: 12 }}>{s.label}: {Math.round(s.value)} kcal • {Math.round((s.value / sum) * 100)}%</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: "32px 20px", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif", background: '#f8f9fa', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* Header Section */}
        <div style={{ marginBottom: '32px', paddingBottom: '24px', borderBottom: '1px solid #e2e8f0' }}>
          <h1 style={{ marginBottom: '8px', fontSize: '32px', fontWeight: '700', color: '#1a202c' }}>
            Nutrition & Fitness Recommender
          </h1>
          <p style={{ color: '#718096', marginBottom: '0', fontSize: '15px', lineHeight: '1.6' }}>
            Get personalized nutrition and fitness recommendations based on your goals and preferences
          </p>
        </div>

        {/* Main Grid Layout */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px', marginBottom: '24px' }}>

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
                  setBmi(data.bmi ?? null);
                  setUserHeight(data.user_height ?? null);
                  setUserWeight(data.user_weight ?? null);
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
            <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px', color: '#1a202c' }}>Load Existing User</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '6px', fontWeight: '500', color: '#4a5568', fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Search Users</label>
                <input 
                  type="text" 
                  placeholder="Search by ID, age, weight..." 
                  value={userSearchQuery} 
                  onChange={(e) => { setUserSearchQuery(e.target.value); setUserPageIndex(0); }}
                  style={{ padding: '10px 12px', borderRadius: '6px', border: '1px solid #e2e8f0', fontSize: '14px', width: '100%' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', marginBottom: '6px', fontWeight: '500', color: '#4a5568', fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Select User</label>
                <select 
                  value={loadUserId} 
                  onChange={(e) => setLoadUserId(e.target.value)}
                  size={6}
                  style={{ padding: '10px 12px', borderRadius: '6px', border: '1px solid #e2e8f0', fontSize: '13px', fontFamily: 'monospace', width: '100%' }}
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
              </div>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <button onClick={async () => { if (!loadUserId) { alert('Select a user'); return; } await loadExistingUser(); }} style={buttonPrimaryStyle} title="Load and fetch recommendations for selected user">Load User</button>
                <button onClick={() => fetchUsers(userPageIndex * usersPerPage)} style={buttonStyle}>Refresh</button>
                <button onClick={() => { if (userPageIndex > 0) fetchUsers((userPageIndex - 1) * usersPerPage); }} style={buttonStyle}>← Prev</button>
                <button onClick={() => { fetchUsers((userPageIndex + 1) * usersPerPage); }} style={buttonStyle}>Next →</button>
                <span style={{ marginLeft: 'auto', alignSelf: 'center', color: '#718096', fontSize: '12px' }}>Page {userPageIndex + 1}</span>
              </div>
              {userCreated && (
                <div style={{ padding: '12px', background: '#c6f6d5', borderRadius: '6px', fontSize: '13px', color: '#22543d', border: '1px solid #9ae6b4' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', justifyContent: 'space-between' }}>
                    <div>✓ Current User ID: <strong>{userId}</strong></div>
                    <button
                      onClick={async () => {
                        if (!window.confirm('Delete this user and their interactions? This cannot be undone.')) return;
                        try {
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
                      style={{ padding: '6px 12px', borderRadius: '4px', background: '#f56565', color: '#fff', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: '500' }}
                    >
                      Delete User
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

        </div>

        {/* Recommendations Section */}
        <div style={cardStyle}>
          <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px', color: '#1a202c' }}>Get Recommendations</h2>
          {!userCreated ? (
            <div style={{ padding: '16px', background: '#fed7d7', borderRadius: '6px', color: '#742a2a', fontSize: '14px', border: '1px solid #fc8181' }}>
              ✓ Create a user first using the form on the left to get started.
            </div>
          ) : (
            <div>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '16px' }}>
                <button type="button" onClick={async () => {
                  try {
                    const res = await fetch(`/api/recommendations/${userId}?top_k=5&rec_type=all`);
                    const data = await res.json();
                    setRecommendations(data);
                    if (data) {
                      setTdee(data.tdee ?? null);
                      setBmi(data.bmi ?? null);
                      setUserHeight(data.user_height ?? null);
                      setUserWeight(data.user_weight ?? null);
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
                      setBmi(data.bmi ?? null);
                      setUserHeight(data.user_height ?? null);
                      setUserWeight(data.user_weight ?? null);
                      setBmr(data.bmr ?? null);
                      if (data.nutrition_targets) setNutritionTargets(data.nutrition_targets);
                      if (data.meal_plan) setMealPlan(data.meal_plan);
                    }
                  } catch (e) {
                    console.debug('Failed to fetch recommendations', e);
                  }
                }} style={{...buttonPrimaryStyle, background: '#d69e2e'}}>Nutrition Only</button>
                
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
                      setBmi(data.bmi ?? null);
                      setUserHeight(data.user_height ?? null);
                      setUserWeight(data.user_weight ?? null);
                      setBmr(data.bmr ?? null);
                    }
                  } catch (e) {
                    console.debug('Failed to fetch recommendations', e);
                  }
                }} style={{...buttonPrimaryStyle, background: '#38a169'}}>Fitness Only</button>
              </div>

              {/* Body Part Filter for Fitness */}
              <div style={{ marginBottom: '16px' }}>
                <label style={{ fontSize: '13px', fontWeight: '600', color: '#4a5568', marginRight: '8px', display: 'block', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Filter by Body Part</label>
                <select 
                  value={selectedBodypart || ''} 
                  onChange={(e) => setSelectedBodypart(e.target.value || null)}
                  style={{
                    padding: '10px 12px',
                    borderRadius: '6px',
                    border: '1px solid #e2e8f0',
                    fontSize: '14px',
                    background: '#fff',
                    cursor: 'pointer',
                    width: '100%'
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

        {/* Unified Stats Card with large daily/weekly numbers and meal distribution pie */}
        {userCreated && tdee && (
          <div style={cardStyle}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '20px', color: '#1a202c' }}>Your Daily Targets</h2>
            <div style={{ display: 'flex', gap: '16px', alignItems: 'stretch', flexWrap: 'wrap' }}>
              {/* Left big calories box */}
              <div style={{ flex: '0 0 auto', width: '280px', background: 'linear-gradient(135deg, #f6e5c3 0%, #fa7e1e 100%)', borderRadius: '12px', padding: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'center', boxShadow: '0 4px 12px rgba(250, 126, 30, 0.2)' }}>
                <div style={{ color: 'rgba(0,0,0,0.6)', fontSize: '13px', marginBottom: '8px', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Daily Calories Goal</div>
                <div style={{ fontSize: '48px', fontWeight: '800', color: '#fff', lineHeight: '1', textShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>{Math.round(tdee)}</div>
                <div style={{ color: 'rgba(255,255,255,0.9)', marginTop: '8px', fontSize: '13px', fontWeight: '500' }}>kcal per day</div>
              </div>

              {/* Right side: small stat cards, activity table and pie chart */}
              <div style={{ flex: '1', minWidth: '360px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'flex', gap: '12px' }}>
                  <div style={{ flex: '1', padding: '16px', borderRadius: '8px', background: 'linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%)', borderLeft: '4px solid #22c55e' }}>
                    <div style={{ fontSize: '12px', color: '#22543d', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '0.5px' }}>BMR</div>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: '#065f46' }}>{bmr ? Math.round(bmr) : '—'}</div>
                    <div style={{ fontSize: '12px', color: '#22543d' }}>kcal/day</div>
                  </div>
                  <div style={{ flex: '1', padding: '16px', borderRadius: '8px', background: 'linear-gradient(135deg, #fed7aa 0%, #fb923c 100%)', borderLeft: '4px solid #f97316' }}>
                    <div style={{ fontSize: '12px', color: '#7c2d12', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Protein Goal</div>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: '#92400e' }}>{nutritionTargets && nutritionTargets.protein_g ? Math.round(nutritionTargets.protein_g) : '—'}</div>
                    <div style={{ fontSize: '12px', color: '#92400e' }}>grams/day</div>
                  </div>
                  <div style={{ flex: '1', padding: '16px', borderRadius: '8px', background: 'linear-gradient(135deg, #cffafe 0%, #67e8f9 100%)', borderLeft: '4px solid #06b6d4' }}>
                    <div style={{ fontSize: '12px', color: '#164e63', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '0.5px' }}>BMI</div>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: '#0c4a6e' }}>{bmi ? bmi.toFixed(1) : '—'}</div>
                    <div style={{ fontSize: '12px', color: '#164e63' }}>kg/m²</div>
                  </div>
                </div>

                {/* BMI Details Section */}
                {(() => {
                  console.log('BMI Section - bmi:', bmi, 'userHeight:', userHeight, 'userForm.height:', userForm?.height);
                  // Use userHeight from API, fallback to form height, then fallback to 170
                  const displayHeight = userHeight || (userForm?.height ? parseFloat(userForm.height) : 170);
                  const showBmiDetails = bmi !== null && bmi !== undefined;
                  console.log('Show BMI Details?', showBmiDetails, 'displayHeight:', displayHeight);
                  return showBmiDetails ? (
                    <div style={{ padding: '24px', borderRadius: '12px', background: '#f5f5f5', border: '1px solid #e0e0e0', marginBottom: '16px' }}>
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '30px' }}>
                        {/* Left: BMI Value & Category */}
                        <div style={{ minWidth: '100px' }}>
                          <div style={{ fontSize: '48px', fontWeight: '800', color: '#1a1a1a' }}>{bmi.toFixed(1)}</div>
                          <div style={{ fontSize: '13px', fontWeight: '600', color: '#0066cc', marginTop: '8px', lineHeight: '1.4' }}>
                            {bmi < 18.5 ? 'Underweight' : bmi < 25 ? 'Normal Weight' : bmi < 30 ? 'Overweight' : 'Obese'}
                          </div>
                          <div style={{ fontSize: '12px', color: '#666', marginTop: '8px', lineHeight: '1.4' }}>
                            Healthy Weight Range:<br />{displayHeight ? `${(18.5 * (displayHeight / 100) ** 2).toFixed(1)} - ${(24.9 * (displayHeight / 100) ** 2).toFixed(1)} kg` : '—'}
                          </div>
                        </div>

                        {/* Right: Classification Chart */}
                        <div style={{ flex: 1 }}>
                          {/* Chart Labels */}
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '12px', fontWeight: '600', color: '#666' }}>
                            <div style={{ textAlign: 'center', flex: 1 }}>Underweight<br />&lt; 18.5</div>
                            <div style={{ textAlign: 'center', flex: 1 }}>Normal<br />18.5-25</div>
                            <div style={{ textAlign: 'center', flex: 1 }}>Overweight<br />25-30</div>
                            <div style={{ textAlign: 'center', flex: 1 }}>Obese<br />&gt; 30</div>
                          </div>

                          {/* Chart Bar with Pointer */}
                          <div style={{ position: 'relative', marginBottom: '8px', paddingTop: '28px' }}>
                            <div style={{ display: 'flex', gap: '2px', height: '32px', borderRadius: '4px', overflow: 'visible' }}>
                              <div style={{ flex: 1, background: '#b3d9ff', border: bmi < 18.5 ? '3px solid #0066cc' : 'none' }}></div>
                              <div style={{ flex: 1, background: '#66b3ff', border: (bmi >= 18.5 && bmi < 25) ? '3px solid #0066cc' : 'none' }}></div>
                              <div style={{ flex: 1, background: '#3366ff', border: (bmi >= 25 && bmi < 30) ? '3px solid #0066cc' : 'none' }}></div>
                              <div style={{ flex: 1, background: '#0033cc', border: bmi >= 30 ? '3px solid #0066cc' : 'none' }}></div>
                            </div>
                            
                            {/* Pointer Arrow positioned at middle of BMI category */}
                            {(() => {
                              // Position pointer at the middle of each BMI category
                              let pointerPercent = 0;
                              if (bmi < 18.5) {
                                // Underweight: middle at 12.5%
                                pointerPercent = 12.5;
                              } else if (bmi < 25) {
                                // Normal: middle at 37.5%
                                pointerPercent = 37.5;
                              } else if (bmi < 30) {
                                // Overweight: middle at 62.5%
                                pointerPercent = 62.5;
                              } else {
                                // Obese: middle at 87.5%
                                pointerPercent = 87.5;
                              }
                              return (
                                <div style={{ 
                                  position: 'absolute', 
                                  left: `calc(${pointerPercent}% - 8px)`,
                                  top: '0',
                                  display: 'flex',
                                  flexDirection: 'column',
                                  alignItems: 'center',
                                  gap: '2px'
                                }}>
                                  <div style={{ fontSize: '11px', fontWeight: '700', color: '#1a1a1a', whiteSpace: 'nowrap' }}>You</div>
                                  <div style={{ width: '0', height: '0', borderLeft: '6px solid transparent', borderRight: '6px solid transparent', borderBottom: '8px solid #1a1a1a' }}></div>
                                </div>
                              );
                            })()}
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : null;
                })()}

                {/* Lower panel: activity table and TDEE chart */}
                <div style={{ display: 'flex', gap: '12px' }}>
                  <div style={{ flex: '1', padding: '16px', borderRadius: '8px', background: '#f8f9fa', border: '1px solid #e2e8f0' }}>
                    <div style={{ fontSize: '13px', fontWeight: '700', color: '#1a202c', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Activity Estimates</div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px' }}>
                      <div style={{ flex: '1', color: '#4a5568' }}>
                        <div style={{ padding: '6px 0', fontSize: '13px' }}>Low</div>
                        <div style={{ padding: '6px 0', fontSize: '13px' }}>Medium</div>
                        <div style={{ padding: '6px 0', fontSize: '13px' }}>High</div>
                      </div>
                      <div style={{ minWidth: '110px', textAlign: 'right', color: '#1a202c', fontWeight: '700', fontSize: '13px' }}>
                        <div style={{ padding: '6px 0' }}>{bmr ? Math.round(bmr * 1.2) : '—'}</div>
                        <div style={{ padding: '6px 0' }}>{bmr ? Math.round(bmr * 1.55) : '—'}</div>
                        <div style={{ padding: '6px 0' }}>{bmr ? Math.round(bmr * 1.725) : '—'}</div>
                      </div>
                    </div>
                  </div>

                  <div style={{ width: '220px', padding: '16px', borderRadius: '8px', background: '#f8f9fa', border: '1px solid #e2e8f0', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <div style={{ fontSize: '13px', fontWeight: '700', color: '#1a202c', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>TDEE Breakdown</div>
                    <div>
                      {bmr && tdee ? (
                        <MealPieChart bmr={bmr} tdee={tdee} size={140} />
                      ) : (
                        <div style={{ color: '#718096', fontSize: '13px' }}>No data</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Recommendations Display */}
        {recommendations ? (
          <div style={cardStyle}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '20px', color: '#1a202c' }}>Recommendations & Meal Plan</h2>
            
            {/* BMI-Based Recommendation Card */}
            {bmi !== null && (
              <div style={{ marginBottom: '20px', padding: '16px', borderRadius: '8px', border: '1px solid #e2e8f0', background: bmi < 18.5 ? '#fef3c7' : bmi < 25 ? '#d1fae5' : bmi < 30 ? '#fed7aa' : '#fecaca' }}>
                <div style={{ fontWeight: '600', fontSize: '14px', marginBottom: '8px', color: bmi < 18.5 ? '#92400e' : bmi < 25 ? '#065f46' : bmi < 30 ? '#92400e' : '#7f1d1d' }}>
                  {bmi < 18.5 ? '⚡ Underweight Recommendation' : bmi < 25 ? '✓ Healthy Weight Recommendation' : bmi < 30 ? '⚠ Overweight Recommendation' : '🚨 Obese Recommendation'}
                </div>
                <div style={{ fontSize: '13px', color: bmi < 18.5 ? '#78350f' : bmi < 25 ? '#15803d' : bmi < 30 ? '#b45309' : '#991b1b', lineHeight: '1.5' }}>
                  {bmi < 18.5 ? (
                    <>
                      Your BMI is below the healthy range. Focus on <strong>increasing calorie intake</strong> with nutrient-dense foods. Aim for <strong>protein-rich meals</strong> to build muscle mass. Include healthy fats (nuts, oils, avocado) and increase portion sizes. Consider strength training combined with adequate nutrition.
                    </>
                  ) : bmi < 25 ? (
                    <>
                      Your BMI is in the healthy range! <strong>Maintain your current habits</strong> and follow the recommended meal plan. Continue with <strong>regular exercise</strong> and balanced nutrition to sustain your health.
                    </>
                  ) : bmi < 30 ? (
                    <>
                      Your BMI is in the overweight range. <strong>Create a modest caloric deficit</strong> through balanced meals and regular exercise. Reduce processed foods and sugar, increase fiber intake, and aim for <strong>150+ minutes of moderate exercise</strong> per week. Follow the recommended meal plan carefully.
                    </>
                  ) : (
                    <>
                      Your BMI indicates obesity. <strong>Consult with a healthcare provider</strong> before making major dietary changes. Focus on <strong>gradual, sustainable lifestyle changes</strong>: increase physical activity gradually, reduce caloric intake moderately, and prioritize whole foods. Professional guidance is recommended.
                    </>
                  )}
                </div>
              </div>
            )}

            {mealPlan && (
              <div style={{ marginBottom: '32px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '12px', color: '#1a202c' }}>Daily Meal Plan</h3>
                {recommendations.daily_protein !== undefined && recommendations.daily_calories !== undefined && (
                  <div style={{ marginBottom: '16px', padding: '12px', background: '#f0f9ff', borderRadius: '6px', border: '1px solid #bae6fd', fontSize: '14px' }}>
                    <strong style={{ color: '#0369a1' }}>Goal:</strong>
                    <span style={{ color: '#1e40af' }}> {Math.round(recommendations.daily_protein || 0)} g protein • {Math.round(recommendations.daily_calories || 0)} kcal</span>
                    {recommendations.daily_fiber !== undefined && recommendations.daily_magnesium !== undefined ?
                      <span style={{ color: '#1e40af' }}> • {Math.round(recommendations.daily_fiber || 0)} g fiber • {Math.round(recommendations.daily_magnesium || 0)} mg magnesium</span> : ''}
                    {recommendations.daily_magnesium_pct ? <span style={{ color: '#1e40af' }}> ({Math.round(recommendations.daily_magnesium_pct)}% RDA)</span> : ''}
                    {recommendations.magnesium_warning ? (
                      <span style={{ color: '#dc2626', marginLeft: '12px', fontWeight: '600' }}>⚠ High magnesium — review servings</span>
                    ) : null}
                  </div>
                )}
                <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                  {Object.keys(mealPlan).map((mealName) => {
                    const items = mealPlan[mealName] || [];
                    const totals = items.reduce(
                      (acc, it) => {
                        acc.calories += Number(it.calories || 0);
                        acc.protein += Number(it.protein || 0);
                        acc.carbs += Number(it.carbohydrates || it.carbs_g || 0);
                        acc.fat += Number(it.fat || 0);
                        acc.fiber += Number(it.fiber || 0);
                        acc.magnesium += Number(it.magnesium || 0);
                        return acc;
                      },
                      { calories: 0, protein: 0, carbs: 0, fat: 0, fiber: 0, magnesium: 0 }
                    );

                    return (
                      <div key={mealName} style={{ minWidth: '280px', flex: '1 1 280px', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '16px', boxShadow: '0 1px 3px rgba(0,0,0,0.06)', background: '#fff' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                          <strong style={{ fontSize: '15px', color: '#1a202c' }}>{mealName}</strong>
                          <small style={{ color: '#718096', fontSize: '12px', fontWeight: '500' }}>{items.length} items</small>
                        </div>
                        <ul style={{ padding: '0', margin: '0', listStyle: 'none' }}>
                          {items.map((it, i) => (
                            <li key={it.id || `${mealName}-${i}`} style={{ marginBottom: '12px', paddingBottom: '12px', borderBottom: '1px dashed #e2e8f0' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '8px' }}>
                                <div>
                                  <div style={{ fontWeight: '600', display: 'flex', alignItems: 'center', gap: '8px', color: '#1a202c', marginBottom: '4px' }}>
                                    {it.food || it.name}
                                    {it.reason === 'liked' && <span style={{ fontSize: '11px', background: '#c6f6d5', color: '#22543d', padding: '2px 8px', borderRadius: '4px', fontWeight: '500' }}>Liked</span>}
                                    {it.reason === 'snack' && <span style={{ fontSize: '11px', background: '#fef08a', color: '#854d0e', padding: '2px 8px', borderRadius: '4px', fontWeight: '500' }}>Snack</span>}
                                  </div>
                                  <div style={{ fontSize: '12px', color: '#718096' }}>{it.category || ''}{it.tags ? ` • ${it.tags}` : ''}</div>
                                </div>
                                <div style={{ textAlign: 'right', fontSize: '12px', color: '#1a202c', fontWeight: '600' }}>
                                  <div>{Math.round(Number(it.calories || 0))} kcal</div>
                                  <div style={{ fontSize: '11px', color: '#718096' }}>{Math.round(Number(it.protein || 0))} g protein</div>
                                  {it.serving_multiplier && it.serving_multiplier > 1.05 && <div style={{ fontSize: '11px', color: '#718096', fontStyle: 'italic' }}>x{it.serving_multiplier.toFixed(1)}</div>}
                                </div>
                              </div>
                              <div style={{ marginBottom: '12px', fontSize: '12px', color: '#4a5568', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                                <span>Carbs: {Math.round(Number(it.carbohydrates || it.carbs_g || 0))} g</span>
                                <span>Fat: {Math.round(Number(it.fat || 0))} g</span>
                                <span>Fiber: {Math.round(Number(it.fiber || 0))} g</span>
                                <span>Mg: {Math.round(Number(it.magnesium || 0))} mg</span>
                              </div>
                              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                                <button onClick={async () => { const nid = it.id || it.item_id; if (nid && userCreated) { setLikedItemIds(prev => new Set(prev).add(nid)); setDislikedItemIds(prev => { const n = new Set(prev); n.delete(nid); return n; }); try { await apiLogInteraction({ user_id: userId, nutrition_item_id: nid, event_type: 'like' }); await getRecommendationsForUser(userId); } catch (e) { console.error('Failed to log like', e); setLikedItemIds(prev => { const n = new Set(prev); n.delete(nid); return n; }); alert('Failed to record like'); } } }} disabled={likedItemIds.has(it.id || it.item_id)} style={{ padding: '6px 12px', borderRadius: '4px', background: likedItemIds.has(it.id || it.item_id) ? '#c6f6d5' : '#fff', border: '1px solid ' + (likedItemIds.has(it.id || it.item_id) ? '#86efac' : '#e2e8f0'), cursor: 'pointer', color: likedItemIds.has(it.id || it.item_id) ? '#22543d' : '#4a5568', fontSize: '12px', fontWeight: '500' }}>{likedItemIds.has(it.id || it.item_id) ? '✓ Liked' : 'Like'}</button>
                                <button onClick={async () => { const nid = it.id || it.item_id; if (nid && userCreated) { setDislikedItemIds(prev => new Set(prev).add(nid)); setLikedItemIds(prev => { const n = new Set(prev); n.delete(nid); return n; }); try { await apiLogInteraction({ user_id: userId, nutrition_item_id: nid, event_type: 'dislike' }); await getRecommendationsForUser(userId); } catch (e) { console.error('Failed to log dislike', e); setDislikedItemIds(prev => { const n = new Set(prev); n.delete(nid); return n; }); alert('Failed to record dislike'); } } }} disabled={dislikedItemIds.has(it.id || it.item_id)} style={{ padding: '6px 12px', borderRadius: '4px', background: dislikedItemIds.has(it.id || it.item_id) ? '#fed7d7' : '#fff', border: '1px solid ' + (dislikedItemIds.has(it.id || it.item_id) ? '#fca5a5' : '#e2e8f0'), cursor: 'pointer', color: dislikedItemIds.has(it.id || it.item_id) ? '#742a2a' : '#4a5568', fontSize: '12px', fontWeight: '500' }}>{dislikedItemIds.has(it.id || it.item_id) ? '✗ Disliked' : 'Dislike'}</button>
                              </div>
                            </li>
                          ))}
                        </ul>
                        <div style={{ marginTop: '12px', borderTop: '1px solid #e2e8f0', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                          <div style={{ fontWeight: '600', color: '#1a202c' }}>Meal total</div>
                          <div style={{ textAlign: 'right' }}>
                            <div style={{ fontWeight: '600', color: '#1a202c' }}>{Math.round(totals.calories)} kcal</div>
                            <div style={{ color: '#718096', fontSize: '12px' }}>{Math.round(totals.protein)} g protein</div>
                            <div style={{ color: '#718096', fontSize: '12px' }}>{Math.round(totals.fiber)} g fiber • {Math.round(totals.magnesium)} mg</div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            {recommendations.fitness_items && recommendations.fitness_items.length > 0 && (
              <div style={{ marginTop: '32px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px', color: '#1a202c' }}>Fitness Recommendations</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '16px' }}>
                  {recommendations.fitness_items.map((activity, idx) => (
                    <div key={activity.id || idx} style={{ border: '1px solid #e2e8f0', borderRadius: '8px', padding: '16px', boxShadow: '0 1px 3px rgba(0,0,0,0.06)', background: '#fff', transition: 'all 0.2s ease', cursor: 'pointer' }} onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = '0 8px 16px rgba(0,0,0,0.12)'; }} onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.06)'; }}>
                      <div style={{ marginBottom: '12px' }}>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: '#1a202c', marginBottom: '8px' }}>{activity.name}</div>
                        <div style={{ fontSize: '12px', color: '#4a5568', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                          <span style={{ background: '#f0f4f8', padding: '4px 10px', borderRadius: '4px', fontWeight: '500' }}>{activity.type}</span>
                          <span style={{ background: '#f0f4f8', padding: '4px 10px', borderRadius: '4px', fontWeight: '500' }}>{activity.level}</span>
                          <span style={{ background: '#f0f4f8', padding: '4px 10px', borderRadius: '4px', fontWeight: '500' }}>{activity.bodypart}</span>
                        </div>
                      </div>
                      {activity.equipment && <div style={{ marginBottom: '12px', fontSize: '13px' }}><span style={{ color: '#718096' }}>Equipment: </span><span style={{ fontWeight: '600', color: '#1a202c' }}>{activity.equipment}</span></div>}
                      {activity.hybrid_score !== undefined && activity.hybrid_score !== null && (
                        <div style={{ marginBottom: '12px', padding: '12px', background: '#f0fdf4', borderRadius: '6px', fontSize: '13px', border: '1px solid #86efac' }}>
                          <div style={{ color: '#4a5568', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: '500' }}>Match Score</div>
                          <div style={{ fontSize: '20px', fontWeight: '700', color: '#22c55e', marginTop: '4px' }}>{Math.round(activity.hybrid_score * 100)}%</div>
                        </div>
                      )}
                      <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                        <button onClick={async () => { const fid = activity.id; if (fid && userCreated) { setLikedItemIds(prev => new Set(prev).add(fid)); setDislikedItemIds(prev => { const n = new Set(prev); n.delete(fid); return n; }); try { await apiLogInteraction({ user_id: userId, fitness_item_id: fid, event_type: 'like' }); await getRecommendationsForUser(userId); } catch (e) { console.error('Failed to log like', e); setLikedItemIds(prev => { const n = new Set(prev); n.delete(fid); return n; }); alert('Failed to record like'); } } }} disabled={likedItemIds.has(activity.id)} style={{ flex: 1, padding: '8px 12px', borderRadius: '4px', background: likedItemIds.has(activity.id) ? '#c6f6d5' : '#fff', border: likedItemIds.has(activity.id) ? '1px solid #86efac' : '1px solid #e2e8f0', cursor: 'pointer', fontSize: '13px', fontWeight: '600', color: likedItemIds.has(activity.id) ? '#22543d' : '#4a5568' }}>{likedItemIds.has(activity.id) ? '✓ Liked' : 'Like'}</button>
                        <button onClick={async () => { const fid = activity.id; if (fid && userCreated) { setDislikedItemIds(prev => new Set(prev).add(fid)); setLikedItemIds(prev => { const n = new Set(prev); n.delete(fid); return n; }); try { await apiLogInteraction({ user_id: userId, fitness_item_id: fid, event_type: 'dislike' }); await getRecommendationsForUser(userId); } catch (e) { console.error('Failed to log dislike', e); setDislikedItemIds(prev => { const n = new Set(prev); n.delete(fid); return n; }); alert('Failed to record dislike'); } } }} disabled={dislikedItemIds.has(activity.id)} style={{ flex: 1, padding: '8px 12px', borderRadius: '4px', background: dislikedItemIds.has(activity.id) ? '#fed7d7' : '#fff', border: dislikedItemIds.has(activity.id) ? '1px solid #fca5a5' : '1px solid #e2e8f0', cursor: 'pointer', fontSize: '13px', fontWeight: '600', color: dislikedItemIds.has(activity.id) ? '#742a2a' : '#4a5568' }}>{dislikedItemIds.has(activity.id) ? '✗ Disliked' : 'Dislike'}</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div style={cardStyle}>
            <p style={{ color: '#718096', fontSize: '14px', margin: '0' }}>Load a user and click "Get Recommendations" to see meal plans and recommendations.</p>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
