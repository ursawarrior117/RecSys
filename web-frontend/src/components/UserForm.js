import React, { useState } from "react";
import { createUser, loadUser, getUsers, fetchRecommendations, deleteAllUsers } from "../services/api";

export default function UserForm({ onResults }) {
  const [createMode, setCreateMode] = useState(true);
  const [loading, setLoading] = useState(false);
  const [users, setUsers] = useState([]);
  const [error, setError] = useState("");
  
  const [createForm, setCreateForm] = useState({
    id: "", name: "", external_id: "", age: 30, weight: 75, height: 175, gender: "M",
    activity_level: "medium", health_goals: "MG", sleep_good: "Yes", dietary_restrictions: ""
  });
  
  const [recForm, setRecForm] = useState({
    user_id: "", item_type: "nutrition", top_k: 10
  });

  const handleCreateChange = (e) => {
    const { name, value } = e.target;
    setCreateForm({ ...createForm, [name]: value });
  };

  const handleRecChange = (e) => {
    const { name, value } = e.target;
    setRecForm({ ...recForm, [name]: value });
  };

  const handleCreateUser = async () => {
    setError("");
    setLoading(true);
    try {
      // Name is optional (backend auto-generates if missing); External ID auto-generated
      const userData = {
        id: createForm.id ? parseInt(createForm.id) : undefined,
        name: createForm.name || undefined,  // send if provided, else let backend auto-generate
        external_id: createForm.external_id || undefined,
        age: parseInt(createForm.age),
        weight: parseFloat(createForm.weight),
        height: parseInt(createForm.height),
        gender: createForm.gender,
        activity_level: createForm.activity_level,
        health_goals: createForm.health_goals,
        sleep_good: createForm.sleep_good === "Yes" ? 1 : 0,
        dietary_restrictions: createForm.dietary_restrictions || undefined
      };
      const newUser = await createUser(userData);
      setError("");
      alert(`User created with ID: ${newUser.id} and Name: ${newUser.name}`);
      setRecForm({ ...recForm, user_id: newUser.id });
      await loadUsersList();
      // notify parent (App) that a user was created so it can set active user state
      try {
        if (typeof onResults === 'function') {
          // no recommendations yet, so pass null for data and uid for user id
          onResults(null, newUser.id);
        }
      } catch (e) {
        // ignore callback failures
      }
      // reset form
      setCreateForm({
        id: "", name: "", external_id: "", age: 30, weight: 75, height: 175, gender: "M",
        activity_level: "medium", health_goals: "MG", sleep_good: "Yes", dietary_restrictions: ""
      });
    } catch (e) {
      setError(`Failed to create user: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleClearAllUsers = async () => {
    if (!window.confirm('Delete ALL users and interactions? This cannot be undone.')) return;
    try {
      setLoading(true);
      const res = await deleteAllUsers();
      alert(`Cleared users: ${res.deleted_users || 0}, interactions: ${res.deleted_interactions || 0}`);
      setUsers([]);
      setRecForm({ ...recForm, user_id: '' });
    } catch (e) {
      setError(`Failed to clear users: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const loadUsersList = async () => {
    try {
      const userList = await getUsers();
      setUsers(userList);
    } catch (e) {
      console.error("Failed to load users:", e);
    }
  };

  const handleLoadUser = async (userId) => {
    setError("");
    setLoading(true);
    try {
      const user = await loadUser(userId);
      setRecForm({ ...recForm, user_id: user.id });
      setError("");
    } catch (e) {
      setError(`Failed to load user: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFetchRecommendations = async () => {
    if (!recForm.user_id) {
      setError("Please select or create a user first");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const payload = { user_id: recForm.user_id, top_k: parseInt(recForm.top_k, 10), item_type: recForm.item_type };
      const data = await fetchRecommendations(payload);
      onResults(data, recForm.user_id);
    } catch (e) {
      setError(`Failed to fetch recommendations: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "0" }}>
      {error && <div style={{ color: "#742a2a", marginBottom: "16px", padding: "12px", background: "#fed7d7", borderRadius: "6px", fontSize: "13px", border: "1px solid #fc8181" }}>{error}</div>}
      
      <div style={{ marginBottom: "24px" }}>
        <h3 style={{ fontSize: "16px", fontWeight: "600", marginBottom: "16px", color: "#1a202c" }}>Create New User</h3>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "12px" }}>
          <div>
            <label style={{ display: "block", marginBottom: "6px", fontWeight: "500", color: "#4a5568", fontSize: "13px", textTransform: "uppercase", letterSpacing: "0.5px" }}>User ID (optional)</label>
            <input type="number" name="id" value={createForm.id} onChange={handleCreateChange} placeholder="Auto-generated" style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", border: "1px solid #e2e8f0", fontSize: "14px" }} />
          </div>
          <div>
            <label style={{ display: "block", marginBottom: "6px", fontWeight: "500", color: "#4a5568", fontSize: "13px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Name (optional)</label>
            <input type="text" name="name" value={createForm.name} onChange={handleCreateChange} placeholder="Alice, Bob, etc." style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", border: "1px solid #e2e8f0", fontSize: "14px" }} />
          </div>
        </div>
        <div style={{ marginBottom: "12px" }}>
          <label style={{ display: "block", marginBottom: "6px", fontWeight: "500", color: "#4a5568", fontSize: "13px", textTransform: "uppercase", letterSpacing: "0.5px" }}>External ID (optional)</label>
          <input type="text" name="external_id" value={createForm.external_id} onChange={handleCreateChange} placeholder="Email or client ID" style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", border: "1px solid #e2e8f0", fontSize: "14px" }} />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px", marginBottom: "12px" }}>
          <div>
            <label style={{ display: "block", marginBottom: "6px", fontWeight: "500", color: "#4a5568", fontSize: "13px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Age</label>
            <input type="number" name="age" value={createForm.age} onChange={handleCreateChange} style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", border: "1px solid #e2e8f0", fontSize: "14px" }} />
          </div>
          <div>
            <label style={{ display: "block", marginBottom: "6px", fontWeight: "500", color: "#4a5568", fontSize: "13px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Weight (kg)</label>
            <input type="number" name="weight" value={createForm.weight} onChange={handleCreateChange} step="0.1" style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", border: "1px solid #e2e8f0", fontSize: "14px" }} />
          </div>
          <div>
            <label style={{ display: "block", marginBottom: "6px", fontWeight: "500", color: "#4a5568", fontSize: "13px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Height (cm)</label>
            <input type="number" name="height" value={createForm.height} onChange={handleCreateChange} style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", border: "1px solid #e2e8f0", fontSize: "14px" }} />
          </div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "12px" }}>
          <div>
            <label style={{ display: "block", marginBottom: "6px", fontWeight: "500", color: "#4a5568", fontSize: "13px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Gender</label>
            <select name="gender" value={createForm.gender} onChange={handleCreateChange} style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", border: "1px solid #e2e8f0", fontSize: "14px" }}>
              <option value="M">Male</option>
              <option value="F">Female</option>
            </select>
          </div>
          <div>
            <label style={{ display: "block", marginBottom: "6px", fontWeight: "500", color: "#4a5568", fontSize: "13px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Activity Level</label>
            <select name="activity_level" value={createForm.activity_level} onChange={handleCreateChange} style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", border: "1px solid #e2e8f0", fontSize: "14px" }}>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "12px" }}>
          <div>
            <label style={{ display: "block", marginBottom: "6px", fontWeight: "500", color: "#4a5568", fontSize: "13px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Health Goals</label>
            <select name="health_goals" value={createForm.health_goals} onChange={handleCreateChange} style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", border: "1px solid #e2e8f0", fontSize: "14px" }}>
              <option value="MG">Muscle Gain</option>
              <option value="WL">Weight Loss</option>
            </select>
          </div>
          <div>
            <label style={{ display: "block", marginBottom: "6px", fontWeight: "500", color: "#4a5568", fontSize: "13px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Sleep Quality</label>
            <select name="sleep_good" value={createForm.sleep_good} onChange={handleCreateChange} style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", border: "1px solid #e2e8f0", fontSize: "14px" }}>
              <option value="Yes">Good</option>
              <option value="No">Poor</option>
            </select>
          </div>
        </div>
        <div style={{ marginBottom: "16px" }}>
          <label style={{ display: "block", marginBottom: "6px", fontWeight: "500", color: "#4a5568", fontSize: "13px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Dietary Restrictions (comma-separated)</label>
          <input 
            type="text" 
            name="dietary_restrictions" 
            value={createForm.dietary_restrictions} 
            onChange={handleCreateChange} 
            placeholder="e.g., vegetarian, gluten-free, dairy-free" 
            style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", border: "1px solid #e2e8f0", fontSize: "14px" }}
          />
        </div>
        <button onClick={handleCreateUser} disabled={loading} style={{ width: "100%", padding: "10px 16px", borderRadius: "6px", border: "none", background: loading ? "#cbd5e0" : "#3182ce", color: "#fff", cursor: loading ? "not-allowed" : "pointer", fontSize: "14px", fontWeight: "600", transition: "all 0.2s ease" }}>
          {loading ? "Creating..." : "Create User"}
        </button>
      </div>

      <div style={{ marginBottom: "24px" }}>
        <h3 style={{ fontSize: "16px", fontWeight: "600", marginBottom: "0", color: "#1a202c" }}>Existing Users</h3>
        <p style={{ fontSize: "13px", color: "#718096", margin: "4px 0 12px 0" }}>Load a user to get personalized recommendations</p>
        <select value={recForm.user_id} onChange={(e) => handleLoadUser(parseInt(e.target.value))} disabled={loading} style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", border: "1px solid #e2e8f0", fontSize: "14px", marginBottom: "12px" }}>
          <option value="">Select user...</option>
          {users.map(u => <option key={u.id} value={u.id}>{`User ${u.id} • Age ${u.age} • ${u.gender}`}</option>)}
        </select>
        <div style={{ display: "flex", gap: "8px" }}>
          <button onClick={loadUsersList} disabled={loading} style={{ flex: 1, padding: "8px 12px", borderRadius: "6px", border: "1px solid #e2e8f0", background: "#fff", cursor: "pointer", fontSize: "13px", fontWeight: "600", color: "#4a5568", transition: "all 0.2s ease" }}>Refresh</button>
          <button style={{ flex: 1, padding: "8px 12px", borderRadius: "6px", background: "#f56565", color: "white", border: "none", cursor: "pointer", fontSize: "13px", fontWeight: "600", transition: "all 0.2s ease" }} onClick={handleClearAllUsers} disabled={loading}>Clear All</button>
        </div>
      </div>
    </div>
  );
}