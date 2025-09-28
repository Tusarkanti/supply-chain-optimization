import React, { useState, useEffect } from 'react';
import ApiService from '../api';
import './LogisticsOptimization.css';

function LogisticsOptimization({ token }) {
  const [routes, setRoutes] = useState([]);
  const [selectedRoute, setSelectedRoute] = useState(null);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Fetch routes from backend
  useEffect(() => {
    fetchRoutes();
  }, [token]);

  const fetchRoutes = async () => {
    try {
      const data = await ApiService.getLogisticsRoutes(token);
      setRoutes(data.routes || []);
    } catch (err) {
      setError('Failed to fetch logistics routes');
    }
  };

  // Handle route selection from dropdown
  const handleSelectRoute = (e) => {
    const routeId = e.target.value;
    const route = routes.find(r => String(r.id) === routeId); // convert to string for comparison
    setSelectedRoute(route || null);
    setOptimizationResult(null); // reset previous optimization
  };

  // Optimize selected route
  const handleOptimize = async () => {
    if (!selectedRoute) return;
    setLoading(true);
    setError('');
    try {
      const data = await ApiService.optimizeRoute(selectedRoute.id, 'cost_and_time', token);
      setOptimizationResult(data);
    } catch (err) {
      console.error(err);
      setError('Failed to optimize route. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Implement optimization
  const handleImplementOptimization = async () => {
    if (!optimizationResult) return;
    try {
      await ApiService.implementOptimization(optimizationResult, token);
      alert('Optimization implemented successfully!');
      setOptimizationResult(null);
      fetchRoutes(); // refresh routes
    } catch (err) {
      alert('Error implementing optimization');
    }
  };

  return (
    <div className="logistics-optimization">
      <div className="logistics-header">
        <h2>Logistics Optimization</h2>
        <p>Optimize your supply chain routes and transportation for maximum efficiency</p>
      </div>

      <div className="optimization-controls">
        <div className="control-group">
          <label htmlFor="route-select">Select Route:</label>
          <select
            id="route-select"
            value={selectedRoute ? String(selectedRoute.id) : ''}
            onChange={handleSelectRoute}
          >
            <option value="">Choose a route...</option>
            {routes.map(route => (
              <option key={route.id} value={String(route.id)}>
                {route.name} - {route.origin} to {route.destination}
              </option>
            ))}
          </select>
        </div>

        <button
          className={`optimize-button ${loading ? 'loading' : ''}`}
          onClick={handleOptimize}
          disabled={!selectedRoute || loading}
        >
          {loading ? 'Optimizing...' : 'Optimize Route'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {selectedRoute && (
        <div className="route-details">
          <h3>Route Details</h3>
          <div className="route-info">
            <div className="info-card">
              <h4>Current Route</h4>
              <p><strong>Origin:</strong> {selectedRoute.origin}</p>
              <p><strong>Destination:</strong> {selectedRoute.destination}</p>
              <p><strong>Distance:</strong> {selectedRoute.distance} km</p>
              <p><strong>Current Cost:</strong> ${selectedRoute.current_cost}</p>
              <p className="eta-display"><strong>ETA:</strong> {selectedRoute.current_time}</p>
            </div>
          </div>
        </div>
      )}

      {optimizationResult && (
        <div className="optimization-results">
          <h3>Optimization Results</h3>
          <div className="results-summary">
            <div className="summary-card">
              <h4>Cost Savings</h4>
              <p className="summary-value">${optimizationResult.cost_savings}</p>
              <p className="summary-percentage">({optimizationResult.cost_savings_percentage}% reduction)</p>
            </div>
            <div className="summary-card">
              <h4>Time Savings</h4>
              <p className="summary-value">{optimizationResult.time_savings} hours</p>
              <p className="summary-percentage">({optimizationResult.time_savings_percentage}% reduction)</p>
            </div>
            <div className="summary-card">
              <h4>CO2 Reduction</h4>
              <p className="summary-value">{optimizationResult.co2_reduction} kg</p>
              <p className="summary-percentage">({optimizationResult.co2_reduction_percentage}% reduction)</p>
            </div>
          </div>

          <div className="optimization-details">
            <h4>Optimized Route Details</h4>
            <div className="details-grid">
              <div className="detail-item">
                <span className="detail-label">New Distance:</span>
                <span className="detail-value">{optimizationResult.new_distance} km</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">New Cost:</span>
                <span className="detail-value">${optimizationResult.new_cost}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">New Time:</span>
                <span className="detail-value">{optimizationResult.new_time} hours</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Optimization Method:</span>
                <span className="detail-value">{optimizationResult.optimization_method}</span>
              </div>
            </div>
          </div>

          <div className="implementation-actions">
            <button className="implement-button" onClick={handleImplementOptimization}>
              Implement Optimization
            </button>
            <button className="cancel-button" onClick={() => setOptimizationResult(null)}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {routes.length === 0 && !loading && (
        <div className="no-routes">
          <p>No logistics routes available for optimization.</p>
        </div>
      )}
    </div>
  );
}

export default LogisticsOptimization;
