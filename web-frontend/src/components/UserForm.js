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
    <div style={{ padding: "20px" }}>
      {error && <div style={{ color: "red", marginBottom: "10px" }}>{error}</div>}
      
      <div style={{ marginBottom: "20px" }}>
        <h3>Create New User</h3>
        <div>
            <label>User ID (optional): </label>
            <input type="number" name="id" value={createForm.id} onChange={handleCreateChange} placeholder="Leave empty for auto-ID" />
        </div>
          <div>
            <label>Name (optional, auto-generated if blank): </label>
            <input type="text" name="name" value={createForm.name} onChange={handleCreateChange} placeholder="e.g., Alice, Bob, etc." />
          </div>
          <div>
            <label>External ID (optional, auto-generated if blank): </label>
            <input type="text" name="external_id" value={createForm.external_id} onChange={handleCreateChange} placeholder="e.g., email or client id" />
          </div>
        <div>
          <label>Age: </label>
          <input type="number" name="age" value={createForm.age} onChange={handleCreateChange} />
        </div>
        <div>
          <label>Weight (kg): </label>
          <input type="number" name="weight" value={createForm.weight} onChange={handleCreateChange} step="0.1" />
        </div>
        <div>
          <label>Height (cm): </label>
          <input type="number" name="height" value={createForm.height} onChange={handleCreateChange} />
        </div>
        <div>
          <label>Gender: </label>
          <select name="gender" value={createForm.gender} onChange={handleCreateChange}>
            <option value="M">Male</option>
            <option value="F">Female</option>
          </select>
        </div>
        <div>
          <label>Activity Level: </label>
          <select name="activity_level" value={createForm.activity_level} onChange={handleCreateChange}>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>
        <div>
          <label>Health Goals: </label>
          <select name="health_goals" value={createForm.health_goals} onChange={handleCreateChange}>
            <option value="MG">Muscle Gain</option>
            <option value="WL">Weight Loss</option>
          </select>
        </div>
        <div>
          <label>Dietary Restrictions (comma-separated): </label>
          <input 
            type="text" 
            name="dietary_restrictions" 
            value={createForm.dietary_restrictions} 
            onChange={handleCreateChange} 
            placeholder="e.g., vegetarian, gluten-free, dairy-free, nut-free, vegan"
          />
        </div>
        <div>
          <label>Sleep Good: </label>
          <select name="sleep_good" value={createForm.sleep_good} onChange={handleCreateChange}>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </div>
        <button onClick={handleCreateUser} disabled={loading}>Create User</button>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <h3>Load Existing User</h3>
        <select value={recForm.user_id} onChange={(e) => handleLoadUser(parseInt(e.target.value))} disabled={loading}>
          <option value="">Select user...</option>
          {users.map(u => <option key={u.id} value={u.id}>{`User ${u.id} - Age ${u.age}, ${u.gender}`}</option>)}
        </select>
        <button onClick={loadUsersList} disabled={loading}>Refresh</button>
        <button style={{ marginLeft: '8px', background: '#d9534f', color: 'white' }} onClick={handleClearAllUsers} disabled={loading}>Clear All Users</button>
      </div>

      {/* Recommendations are fetched via the main App controls — remove duplicate UI here. */}
    </div>
  );
}