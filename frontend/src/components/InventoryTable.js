import React, { useState, useEffect } from 'react';
import './InventoryTable.css';

function InventoryTable({ token }) {
  const [inventoryData, setInventoryData] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showRecommendations, setShowRecommendations] = useState(false);

  useEffect(() => {
    fetchInventoryData();
  }, [token]);

  const fetchInventoryData = async () => {
    setLoading(true);
    setError(null);

    try {
      // This would typically fetch from the backend
      // For now, using mock data
      const mockData = [
        {
          id: 1,
          product_name: 'Product A',
          sku: 'PROD-A-001',
          current_stock: 150,
          reserved_stock: 20,
          available_stock: 130,
          reorder_point: 50,
          max_stock: 200,
          location: 'Warehouse 1'
        },
        {
          id: 2,
          product_name: 'Product B',
          sku: 'PROD-B-002',
          current_stock: 25,
          reserved_stock: 5,
          available_stock: 20,
          reorder_point: 30,
          max_stock: 100,
          location: 'Warehouse 1'
        },
        {
          id: 3,
          product_name: 'Product C',
          sku: 'PROD-C-003',
          current_stock: 0,
          reserved_stock: 0,
          available_stock: 0,
          reorder_point: 10,
          max_stock: 50,
          location: 'Warehouse 2'
        }
      ];

      setInventoryData(mockData);
    } catch (err) {
      setError('Failed to fetch inventory data');
    } finally {
      setLoading(false);
    }
  };

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'https://supply-chain-optimization-2.onrender.com'}/api/inventory/analyze`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          inventory_data_path: 'data/inventory.csv',
          sales_data_path: 'data/sales.csv'
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setRecommendations(data.analysis.recommendations || []);
        setShowRecommendations(true);
      } else {
        setError(data.error || 'Failed to get recommendations');
      }
    } catch (err) {
      setError('Network error');
    } finally {
      setLoading(false);
    }
  };

  const getStockStatus = (item) => {
    if (item.available_stock <= 0) return 'out-of-stock';
    if (item.available_stock <= item.reorder_point) return 'low-stock';
    if (item.available_stock >= item.max_stock * 0.9) return 'overstock';
    return 'normal';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'out-of-stock': return '#ff4444';
      case 'low-stock': return '#ffaa00';
      case 'overstock': return '#ff6b6b';
      default: return '#4caf50';
    }
  };

  if (loading) {
    return <div className="inventory-loading">Loading inventory data...</div>;
  }

  return (
    <div className="inventory-table">
      <div className="inventory-header">
        <h2>Inventory Management</h2>
        <div className="inventory-actions">
          <button onClick={fetchInventoryData} className="refresh-btn">
            Refresh Data
          </button>
          <button onClick={fetchRecommendations} className="recommendations-btn">
            Get Recommendations
          </button>
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="inventory-stats">
        <div className="stat-card">
          <h3>Total Products</h3>
          <span className="stat-value">{inventoryData.length}</span>
        </div>
        <div className="stat-card">
          <h3>Low Stock Items</h3>
          <span className="stat-value low-stock">
            {inventoryData.filter(item => getStockStatus(item) === 'low-stock').length}
          </span>
        </div>
        <div className="stat-card">
          <h3>Out of Stock</h3>
          <span className="stat-value out-of-stock">
            {inventoryData.filter(item => getStockStatus(item) === 'out-of-stock').length}
          </span>
        </div>
        <div className="stat-card">
          <h3>Optimal Stock</h3>
          <span className="stat-value optimal">
            {inventoryData.filter(item => getStockStatus(item) === 'normal').length}
          </span>
        </div>
      </div>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Product</th>
              <th>SKU</th>
              <th>Location</th>
              <th>Current Stock</th>
              <th>Available</th>
              <th>Reserved</th>
              <th>Reorder Point</th>
              <th>Max Stock</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {inventoryData.map((item) => {
              const status = getStockStatus(item);
              return (
                <tr key={item.id} className={`status-${status}`}>
                  <td>{item.product_name}</td>
                  <td>{item.sku}</td>
                  <td>{item.location}</td>
                  <td>{item.current_stock}</td>
                  <td>{item.available_stock}</td>
                  <td>{item.reserved_stock}</td>
                  <td>{item.reorder_point}</td>
                  <td>{item.max_stock}</td>
                  <td>
                    <span
                      className="status-badge"
                      style={{ backgroundColor: getStatusColor(status) }}
                    >
                      {status.replace('-', ' ').toUpperCase()}
                    </span>
                  </td>
                  <td>
                    <button className="action-btn">View Details</button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {showRecommendations && recommendations.length > 0 && (
        <div className="recommendations-section">
          <h3>Inventory Recommendations</h3>
          <div className="recommendations-list">
            {recommendations.map((rec, index) => (
              <div key={index} className="recommendation-card">
                <div className="recommendation-header">
                  <h4>{rec.product_id}</h4>
                  <span className={`priority-${rec.priority.toLowerCase()}`}>
                    {rec.priority} PRIORITY
                  </span>
                </div>
                <div className="recommendation-details">
                  <p><strong>Current Stock:</strong> {rec.current_stock}</p>
                  <p><strong>Recommended Order:</strong> {rec.recommended_order_quantity}</p>
                  <p><strong>Reorder Point:</strong> {rec.reorder_point}</p>
                  <p><strong>Safety Stock:</strong> {rec.safety_stock}</p>
                  <p><strong>Confidence:</strong> {(rec.confidence_score * 100).toFixed(1)}%</p>
                </div>
                <div className="recommendation-reasoning">
                  <p>{rec.reasoning}</p>
                </div>
                <div className="recommendation-actions">
                  <button className="approve-btn">Approve Order</button>
                  <button className="adjust-btn">Adjust Parameters</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default InventoryTable;
