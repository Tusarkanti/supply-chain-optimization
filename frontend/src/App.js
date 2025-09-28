// src/App.js
import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import DemandForecasting from './components/DemandForecasting';
import InventoryManagement from './components/InventoryManagement';
import LogisticsOptimization from './components/LogisticsOptimization';
import Reporting from './components/Reporting';
import Login from './components/Login';
import Chatbot from './components/Chatbot';
import ErrorBoundary from './ErrorBoundary';
import './App.css';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [token, setToken] = useState(null);

  useEffect(() => {
    // Check for saved token on mount
    const savedToken = localStorage.getItem('token');
    if (savedToken) {
      setToken(savedToken);
      setIsAuthenticated(true);
    }
  }, []);

  const handleLogin = (newToken, user = null) => {
    // Store token and user data on login
    setToken(newToken);
    setIsAuthenticated(true);
    localStorage.setItem('token', newToken);
    if (user) {
      localStorage.setItem('user', JSON.stringify(user));
    }
  };

  const handleLogout = () => {
    // Clear authentication state and localStorage
    setToken(null);
    setIsAuthenticated(false);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
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
          <button
            className={activeTab === 'reporting' ? 'active' : ''}
            onClick={() => setActiveTab('reporting')}
          >
            Reports
          </button>
          <button className="logout-btn" onClick={handleLogout}>
            Logout
          </button>
        </nav>
      </header>

      <main className="app-main">
        <ErrorBoundary>
          {activeTab === 'dashboard' && <Dashboard token={token} />}
          {activeTab === 'forecast' && <DemandForecasting token={token} />}
          {activeTab === 'inventory' && <InventoryManagement token={token} />}
          {activeTab === 'logistics' && <LogisticsOptimization token={token} />}
          {activeTab === 'reporting' && <Reporting token={token} />}
        </ErrorBoundary>
      </main>

      <Chatbot />

      <footer className="app-footer">
        <p>&copy; 2025 Supply Chain Optimization System. All rights reserved.</p>
        <p>DEDICATED TO THE DEAREST FRD BUDDI 🥰</p>
      </footer>
    </div>
  );
}

export default App;
