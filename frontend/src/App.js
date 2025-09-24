import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import ForecastForm from './components/ForecastForm';
import InventoryTable from './components/InventoryTable';
import Login from './components/Login';
import './App.css';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [token, setToken] = useState(null);

  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    if (savedToken) {
      setToken(savedToken);
      setIsAuthenticated(true);
    }
  }, []);

  const handleLogin = (newToken) => {
    setToken(newToken);
    setIsAuthenticated(true);
    localStorage.setItem('token', newToken);
  };

  const handleLogout = () => {
    setToken(null);
    setIsAuthenticated(false);
    localStorage.removeItem('token');
  };

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="App">
      <header className="app-header">
        <h1>Intelligent Supply Chain Optimization System</h1>
        <nav className="main-nav">
          <button
            className={activeTab === 'dashboard' ? 'active' : ''}
            onClick={() => setActiveTab('dashboard')}
          >
            Dashboard
          </button>
          <button
            className={activeTab === 'forecast' ? 'active' : ''}
            onClick={() => setActiveTab('forecast')}
          >
            Demand Forecasting
          </button>
          <button
            className={activeTab === 'inventory' ? 'active' : ''}
            onClick={() => setActiveTab('inventory')}
          >
            Inventory Management
          </button>
          <button
            className={activeTab === 'logistics' ? 'active' : ''}
            onClick={() => setActiveTab('logistics')}
          >
            Logistics Optimization
          </button>
          <button className="logout-btn" onClick={handleLogout}>
            Logout
          </button>
        </nav>
      </header>

      <main className="app-main">
        {activeTab === 'dashboard' && <Dashboard token={token} />}
        {activeTab === 'forecast' && <ForecastForm token={token} />}
        {activeTab === 'inventory' && <InventoryTable token={token} />}
        {activeTab === 'logistics' && (
          <div className="logistics-section">
            <h2>Logistics Optimization</h2>
            <p>Route optimization functionality coming soon...</p>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>&copy; 2024 Supply Chain Optimization System. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
