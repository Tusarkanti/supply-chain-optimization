import React, { useState, useEffect } from 'react';
import './Dashboard.css';

function Dashboard({ token }) {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchDashboardData();
  }, [token]);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/api/dashboard/data', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        setDashboardData(data.data);
      } else {
        setError('Failed to fetch dashboard data');
      }
    } catch (err) {
      setError('Network error');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="dashboard-loading">Loading dashboard...</div>;
  }

  if (error) {
    return <div className="dashboard-error">Error: {error}</div>;
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>System Overview</h2>
        <button onClick={fetchDashboardData} className="refresh-btn">
          Refresh
        </button>
      </div>

      <div className="dashboard-grid">
        {/* Demand Forecasting Section */}
        <div className="dashboard-card">
          <h3>Demand Forecasting</h3>
          <div className="card-content">
            <div className="metric">
              <span className="metric-label">Total Products</span>
              <span className="metric-value">25</span>
            </div>
            <div className="metric">
              <span className="metric-label">Active Models</span>
              <span className="metric-value">23</span>
            </div>
            <div className="metric">
              <span className="metric-label">Avg Accuracy</span>
              <span className="metric-value">94.2%</span>
            </div>
          </div>
        </div>

        {/* Inventory Management Section */}
        <div className="dashboard-card">
          <h3>Inventory Status</h3>
          <div className="card-content">
            <div className="metric">
              <span className="metric-label">Total Products</span>
              <span className="metric-value">1,247</span>
            </div>
            <div className="metric">
              <span className="metric-label">Low Stock Alerts</span>
              <span className="metric-value critical">12</span>
            </div>
            <div className="metric">
              <span className="metric-label">Optimal Stock</span>
              <span className="metric-value">89%</span>
            </div>
          </div>
        </div>

        {/* Logistics Optimization Section */}
        <div className="dashboard-card">
          <h3>Logistics Performance</h3>
          <div className="card-content">
            <div className="metric">
              <span className="metric-label">Routes Optimized</span>
              <span className="metric-value">156</span>
            </div>
            <div className="metric">
              <span className="metric-label">Total Distance</span>
              <span className="metric-value">2,847 km</span>
            </div>
            <div className="metric">
              <span className="metric-label">Cost Savings</span>
              <span className="metric-value">23.5%</span>
            </div>
          </div>
        </div>

        {/* Anomaly Detection Section */}
        <div className="dashboard-card">
          <h3>Anomaly Detection</h3>
          <div className="card-content">
            <div className="metric">
              <span className="metric-label">Data Points Analyzed</span>
              <span className="metric-value">45,231</span>
            </div>
            <div className="metric">
              <span className="metric-label">Anomalies Detected</span>
              <span className="metric-value warning">8</span>
            </div>
            <div className="metric">
              <span className="metric-label">Detection Rate</span>
              <span className="metric-value">99.1%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="dashboard-card full-width">
        <h3>Recent Activity</h3>
        <div className="activity-list">
          <div className="activity-item">
            <span className="activity-time">2 hours ago</span>
            <span className="activity-description">Demand forecast updated for Product A</span>
          </div>
          <div className="activity-item">
            <span className="activity-time">4 hours ago</span>
            <span className="activity-description">Inventory optimization completed</span>
          </div>
          <div className="activity-item">
            <span className="activity-time">6 hours ago</span>
            <span className="activity-description">Route optimization saved 15% distance</span>
          </div>
          <div className="activity-item">
            <span className="activity-time">8 hours ago</span>
            <span className="activity-description">Anomaly detected in sales data</span>
          </div>
        </div>
      </div>

      {/* System Health */}
      <div className="dashboard-card">
        <h3>System Health</h3>
        <div className="health-indicators">
          <div className="health-item">
            <span className="health-label">API Status</span>
            <span className="health-status healthy">● Healthy</span>
          </div>
          <div className="health-item">
            <span className="health-label">Database</span>
            <span className="health-status healthy">● Connected</span>
          </div>
          <div className="health-item">
            <span className="health-label">ML Models</span>
            <span className="health-status healthy">● Active</span>
          </div>
          <div className="health-item">
            <span className="health-label">Last Update</span>
            <span className="health-status">{new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
