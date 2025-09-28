import React, { useState, useEffect } from 'react';
import ApiService from '../api';
import './InventoryManagement.css';

function InventoryManagement({ token }) {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [expandedCard, setExpandedCard] = useState(null);
  const [activeTab, setActiveTab] = useState('recommendations');
  const [multiEchelonData, setMultiEchelonData] = useState({
    warehouses: [],
    products: [],
    demand_forecasts: {},
    supplier_data: {}
  });

  useEffect(() => {
    if (activeTab === 'optimization' && multiEchelonData.warehouses.length === 0) {
      // Populate with sample data for multi-echelon optimization
      setMultiEchelonData({
        warehouses: [
          {
            id: 'WH001',
            name: 'Main Warehouse',
            location: { lat: 40.7128, lng: -74.0060 },
            capacity: 10000,
            holding_cost: 0.1
          },
          {
            id: 'WH002',
            name: 'Secondary Warehouse',
            location: { lat: 34.0522, lng: -118.2437 },
            capacity: 5000,
            holding_cost: 0.15
          }
        ],
        products: [
          {
            id: 'PROD001',
            name: 'Product A',
            demand_rate: 50,
            cost: 100,
            holding_cost: 10,
            lead_time: 5
          },
          {
            id: 'PROD002',
            name: 'Product B',
            demand_rate: 30,
            cost: 150,
            holding_cost: 12,
            lead_time: 7
          }
        ],
        demand_forecasts: {
          PROD001: { WH001: 2000, WH002: 1000 },
          PROD002: { WH001: 1500, WH002: 800 }
        },
        supplier_data: {
          SUP001: { lead_time: 3, cost: 90 },
          SUP002: { lead_time: 4, cost: 120 }
        }
      });
    }
  }, [activeTab]);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [alertEmails, setAlertEmails] = useState('');
  const [supplierPerformance, setSupplierPerformance] = useState([]);

  // Futuristic features state
  const [futuristicMetrics, setFuturisticMetrics] = useState(null);
  const [sustainabilityResult, setSustainabilityResult] = useState(null);
  const [riskAssessment, setRiskAssessment] = useState(null);
  const [futuristicLoading, setFuturisticLoading] = useState(false);

  useEffect(() => {
    fetchRecommendations();
    if (activeTab === 'futuristic') {
      fetchFuturisticMetrics();
    }
  }, [token, activeTab]);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError('');
    try {
      // Use futuristic autonomous decisions for enhanced recommendations
      const decisionData = {
        current_state: {
          inventory_levels: {
            PROD001: { current_stock: 45, predicted_demand: 60, reorder_point: 50 },
            PROD002: { current_stock: 120, predicted_demand: 150, reorder_point: 100 },
            PROD003: { current_stock: 20, predicted_demand: 35, reorder_point: 30 }
          }
        },
        historical_performance: {
          stockout_rate: 0.05,
          service_level: 0.92
        }
      };
      const response = await ApiService.makeAutonomousDecisions(decisionData, token);
      if (response.success) {
        // Transform autonomous decisions into recommendation format
        const transformedRecommendations = response.autonomous_decisions.map(decision => ({
          product_id: decision.product_id,
          product_name: decision.product_name || decision.product_id,
          current_stock: decision.current_stock || 0,
          recommended_order_quantity: decision.recommended_order_quantity || 0,
          reorder_point: decision.reorder_point || 0,
          safety_stock: decision.safety_stock || 0,
          confidence_score: decision.confidence || 0.85,
          reasoning: decision.reason || '',
          warehouse_id: decision.warehouse_id || 'WH001',
          lead_time_days: decision.lead_time_days || 7
        }));
        setRecommendations(transformedRecommendations);
      } else {
        // Fallback to basic analysis
        const data = await ApiService.getInventoryAnalysis(token);
        setRecommendations(data.recommendations || []);
      }
    } catch (err) {
      setError('Failed to fetch inventory recommendations');
    } finally {
      setLoading(false);
    }
  };

  const fetchFuturisticMetrics = async () => {
    setFuturisticLoading(true);
    setError('');
    try {
      const response = await ApiService.getFuturisticMetrics(token);
      if (response.success) {
        setFuturisticMetrics(response.metrics);
      }
    } catch (err) {
      setError('Failed to fetch futuristic metrics');
    } finally {
      setFuturisticLoading(false);
    }
  };

  const handleSustainabilityOptimization = async () => {
    setFuturisticLoading(true);
    setError('');
    try {
      const optimizationData = {
        inventory_decisions: {
          PROD001: { transport_distance_km: 100, warehouse_days: 30, manufacturing_intensity: 1.0 },
          PROD002: { transport_distance_km: 150, warehouse_days: 45, manufacturing_intensity: 1.2 }
        },
        environmental_data: { carbon_emissions: 100, waste_reduction: 15 }
      };
      const response = await ApiService.optimizeSustainability(optimizationData, token);
      if (response.success) {
        setSustainabilityResult(response.sustainability_optimization);
      }
    } catch (err) {
      setError('Failed to optimize sustainability');
    } finally {
      setFuturisticLoading(false);
    }
  };

  const handleRiskAssessment = async () => {
    setFuturisticLoading(true);
    setError('');
    try {
      const riskData = {
        inventory_data: { average_daily_demand: 100, lead_time_days: 7 },
        supplier_data: { reliability_score: 0.9 }
      };
      const response = await ApiService.assessSupplyChainRisks(riskData, token);
      if (response.success) {
        setRiskAssessment(response.risk_assessment);
      }
    } catch (err) {
      setError('Failed to assess supply chain risks');
    } finally {
      setFuturisticLoading(false);
    }
  };

  const handleReorder = async (rec) => {
    try {
      await ApiService.placeOrder({
        product_id: rec.product_id,
        warehouse_id: rec.warehouse_id,
        quantity: rec.recommended_order_quantity,
      }, token);
      alert(`Order for ${rec.product_id} placed successfully!`);
      fetchRecommendations();
    } catch (err) {
      alert('Error placing order');
    }
  };

  const getPriorityColor = (reasoning) => {
    if (reasoning.toLowerCase().includes('critical')) return 'critical';
    if (reasoning.toLowerCase().includes('high')) return 'high';
    return 'medium';
  };

  const handleMultiEchelonOptimize = async () => {
    setLoading(true);
    setError('');
    try {
      const result = await ApiService.multiEchelonOptimize(multiEchelonData, token);
      setOptimizationResult(result.optimization_result);
      setActiveTab('optimization');
    } catch (err) {
      setError('Failed to perform multi-echelon optimization');
    } finally {
      setLoading(false);
    }
  };

  const handleSendReorderAlert = async (recommendation) => {
    if (!alertEmails.trim()) {
      alert('Please enter recipient email addresses');
      return;
    }

    try {
      await ApiService.sendReorderAlert({
        recommendation: recommendation,
        recipient_emails: alertEmails.split(',').map(email => email.trim())
      }, token);
      alert('Reorder alert sent successfully!');
    } catch (err) {
      alert('Failed to send reorder alert');
    }
  };

  const fetchSupplierPerformance = async () => {
    // Mock supplier performance data - in real implementation, this would come from API
    setSupplierPerformance([
      {
        supplier_id: 'SUP001',
        name: 'Global Electronics Ltd',
        on_time_delivery: 95.2,
        quality_score: 92.8,
        cost_efficiency: 88.5,
        overall_rating: 92.2
      },
      {
        supplier_id: 'SUP002',
        name: 'Tech Components Inc',
        on_time_delivery: 87.3,
        quality_score: 94.1,
        cost_efficiency: 91.2,
        overall_rating: 90.9
      }
    ]);
  };

  useEffect(() => {
    if (activeTab === 'suppliers') {
      fetchSupplierPerformance();
    }
  }, [activeTab]);

  return (
    <div className="inventory-management">
      <header className="inventory-header">
        <h2>AI-Powered Inventory Management</h2>
        <p>Optimize stock levels with predictive analytics and intelligent reorder recommendations</p>
      </header>

      <div className="tab-navigation">
        <button
          className={`tab-button ${activeTab === 'recommendations' ? 'active' : ''}`}
          onClick={() => setActiveTab('recommendations')}
        >
          Recommendations
        </button>
        <button
          className={`tab-button ${activeTab === 'optimization' ? 'active' : ''}`}
          onClick={() => setActiveTab('optimization')}
        >
          Multi-Echelon Optimization
        </button>
        <button
          className={`tab-button ${activeTab === 'suppliers' ? 'active' : ''}`}
          onClick={() => setActiveTab('suppliers')}
        >
          Supplier Performance
        </button>
        <button
          className={`tab-button ${activeTab === 'futuristic' ? 'active' : ''}`}
          onClick={() => setActiveTab('futuristic')}
        >
          Futuristic Insights
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {activeTab === 'recommendations' && (
        <>
          <div className="inventory-controls">
            <button
              className="refresh-button"
              onClick={fetchRecommendations}
              disabled={loading}
            >
              {loading ? 'Refreshing...' : 'Refresh Recommendations'}
            </button>
          </div>

          <div className="alert-config">
            <h3>Reorder Alert Configuration</h3>
            <div className="alert-input">
              <label>Recipient Email Addresses (comma-separated):</label>
              <input
                type="text"
                value={alertEmails}
                onChange={(e) => setAlertEmails(e.target.value)}
                placeholder="manager@company.com, procurement@company.com"
              />
            </div>
          </div>

          <div className="recommendations-grid">
            {recommendations.map((rec, idx) => (
              <div key={idx} className={`recommendation-card ${getPriorityColor(rec.reasoning)}`}>
                <div className="card-header">
                  <h3>{rec.product_name || rec.product_id}</h3>
                  <span className={`priority-badge ${getPriorityColor(rec.reasoning)}`}>
                    {getPriorityColor(rec.reasoning).toUpperCase()}
                  </span>
                </div>

                <div className="card-content">
                  <div className="inventory-metrics">
                    <div className="metric">
                      <span className="metric-label">Current Stock:</span>
                      <span className="metric-value">{rec.current_stock}</span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Recommended Order:</span>
                      <span className="metric-value">{rec.recommended_order_quantity}</span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Reorder Point:</span>
                      <span className="metric-value">{rec.reorder_point}</span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Safety Stock:</span>
                      <span className="metric-value">{rec.safety_stock}</span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Confidence:</span>
                      <span className="metric-value">{(rec.confidence_score * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  <div className="reasoning">
                    <p>{rec.reasoning}</p>
                  </div>

                  <div className="card-actions">
                    <button
                      className="action-button"
                      onClick={() => handleReorder(rec)}
                    >
                      Place Order
                    </button>
                    <button
                      className="alert-button"
                      onClick={() => handleSendReorderAlert(rec)}
                    >
                      Send Alert
                    </button>
                    <button
                      className="details-button"
                      onClick={() => setExpandedCard(expandedCard === idx ? null : idx)}
                    >
                      {expandedCard === idx ? 'Hide Details' : 'Show Details'}
                    </button>
                  </div>

                  {expandedCard === idx && (
                    <div className="detailed-info">
                      <h4>Additional Information</h4>
                      <p><strong>Warehouse:</strong> {rec.warehouse_id}</p>
                      <p><strong>Lead Time:</strong> {rec.lead_time_days} days</p>
                      <p><strong>Analysis:</strong> Based on historical demand patterns, stock levels, and AI predictive modeling</p>
                      {rec.supplier_recommendations && (
                        <div>
                          <h5>Supplier Recommendations:</h5>
                          <ul>
                            {rec.supplier_recommendations.map((supplier, sidx) => (
                              <li key={sidx}>{supplier.name} - Score: {supplier.score}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {recommendations.length === 0 && !loading && (
            <div className="no-recommendations">
              <p>No inventory recommendations available at this time.</p>
            </div>
          )}
        </>
      )}

      {activeTab === 'optimization' && (
        <div className="optimization-section">
          <h3>Multi-Echelon Inventory Optimization</h3>
          <p>Optimize inventory across multiple warehouse tiers using advanced mathematical modeling</p>

          <div className="optimization-controls">
            <button
              className="optimize-button"
              onClick={handleMultiEchelonOptimize}
              disabled={loading}
            >
              {loading ? 'Optimizing...' : 'Run Multi-Echelon Optimization'}
            </button>
          </div>

          {optimizationResult && (
            <div className="optimization-results">
              <h4>Optimization Results</h4>
              <div className="results-grid">
                <div className="result-card">
                  <h5>Total Cost</h5>
                  <p className="result-value">${optimizationResult.total_cost?.toLocaleString() || 'N/A'}</p>
                </div>
                <div className="result-card">
                  <h5>Service Level</h5>
                  <p className="result-value">{optimizationResult.service_level ? (optimizationResult.service_level * 100).toFixed(1) + '%' : 'N/A'}</p>
                </div>
                <div className="result-card">
                  <h5>Inventory Turnover</h5>
                  <p className="result-value">{optimizationResult.inventory_turnover?.toFixed(2) || 'N/A'}</p>
                </div>
              </div>

              {optimizationResult.recommendations && (
                <div className="recommendations-section">
                  <h5>Recommendations</h5>
                  <ul>
                    {optimizationResult.recommendations.map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === 'suppliers' && (
        <div className="suppliers-section">
          <h3>Supplier Performance Tracking</h3>
          <p>Monitor supplier performance with comprehensive KPIs and scorecards</p>

          <div className="suppliers-grid">
            {supplierPerformance.map((supplier) => (
              <div key={supplier.supplier_id} className="supplier-card">
                <div className="supplier-header">
                  <h4>{supplier.name}</h4>
                  <span className="supplier-id">{supplier.supplier_id}</span>
                </div>

                <div className="supplier-metrics">
                  <div className="metric">
                    <span className="metric-label">On-Time Delivery:</span>
                    <span className="metric-value">{supplier.on_time_delivery}%</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Quality Score:</span>
                    <span className="metric-value">{supplier.quality_score}%</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Cost Efficiency:</span>
                    <span className="metric-value">{supplier.cost_efficiency}%</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Overall Rating:</span>
                    <span className="metric-value overall">{supplier.overall_rating}%</span>
                  </div>
                </div>

                <div className="supplier-actions">
                  <button className="view-details-button">View Details</button>
                  <button className="contact-button">Contact Supplier</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'futuristic' && (
        <div className="futuristic-section">
          <h3>Futuristic Inventory Insights</h3>
          <p>Advanced AI-driven analytics, sustainability optimization, and risk assessment</p>

          <div className="futuristic-controls">
            <button
              className="refresh-button"
              onClick={fetchFuturisticMetrics}
              disabled={futuristicLoading}
            >
              {futuristicLoading ? 'Loading...' : 'Refresh Metrics'}
            </button>
            <button
              className="optimize-button"
              onClick={handleSustainabilityOptimization}
              disabled={futuristicLoading}
            >
              Optimize Sustainability
            </button>
            <button
              className="risk-button"
              onClick={handleRiskAssessment}
              disabled={futuristicLoading}
            >
              Assess Risks
            </button>
          </div>

          {error && <div className="error-message">{error}</div>}

          {futuristicMetrics && (
            <div className="metrics-dashboard">
              <h4>Key Performance Metrics</h4>
              <div className="metrics-grid">
                <div className="metric-card predictive">
                  <h5>Predictive Accuracy</h5>
                  <p className="metric-value">{(futuristicMetrics.predictive_accuracy * 100).toFixed(1)}%</p>
                </div>
                <div className="metric-card sustainability">
                  <h5>Sustainability Score</h5>
                  <p className="metric-value">{futuristicMetrics.sustainability_score.toFixed(1)}</p>
                </div>
                <div className="metric-card risk">
                  <h5>Risk Resilience Index</h5>
                  <p className="metric-value">{futuristicMetrics.risk_resilience_index.toFixed(1)}</p>
                </div>
                <div className="metric-card autonomous">
                  <h5>Autonomous Decisions</h5>
                  <p className="metric-value">{futuristicMetrics.autonomous_decisions}</p>
                </div>
                <div className="metric-card iot">
                  <h5>IoT Sensors Active</h5>
                  <p className="metric-value">{futuristicMetrics.iot_sensors_active}</p>
                </div>
                <div className="metric-card carbon">
                  <h5>Carbon Reduction</h5>
                  <p className="metric-value">{futuristicMetrics.carbon_footprint_reduction.toFixed(1)}%</p>
                </div>
              </div>
            </div>
          )}

          {sustainabilityResult && (
            <div className="sustainability-results">
              <h4>Sustainability Optimization Results</h4>
              <div className="result-highlight">
                <p><strong>Carbon Footprint Reduction:</strong> {sustainabilityResult.carbon_footprint_reduction.toFixed(1)}%</p>
                <p><strong>Sustainability Score:</strong> {sustainabilityResult.sustainability_score.toFixed(1)}</p>
              </div>
              <div className="green-suppliers">
                <h5>Recommended Green Suppliers</h5>
                {sustainabilityResult.green_alternatives.recommended_suppliers.map((supplier, idx) => (
                  <div key={idx} className="supplier-option">
                    <span>{supplier.name}</span>
                    <span className="supplier-rating">{supplier.carbon_rating}</span>
                    <span className="cost-premium">+{supplier.cost_premium}% cost</span>
                  </div>
                ))}
              </div>
              <div className="recommendations-list">
                <h5>Recommendations</h5>
                <ul>
                  {sustainabilityResult.recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {riskAssessment && (
            <div className="risk-assessment">
              <h4>Supply Chain Risk Assessment</h4>
              <div className="risk-score">
                <p><strong>Overall Risk Score:</strong> {riskAssessment.overall_risk_score.toFixed(2)}</p>
              </div>
              <div className="scenario-impacts">
                <h5>Risk Scenarios</h5>
                {Object.entries(riskAssessment.scenario_impacts).map(([scenario, impact]) => (
                  <div key={scenario} className="scenario-card">
                    <h6>{scenario.replace('_', ' ').toUpperCase()}</h6>
                    <p><strong>Risk Score:</strong> {impact.risk_score.toFixed(2)}</p>
                    <p><strong>Impact:</strong> {impact.description}</p>
                  </div>
                ))}
              </div>
              <div className="mitigation-strategies">
                <h5>Mitigation Strategies</h5>
                {Object.entries(riskAssessment.mitigation_strategies).map(([key, strategies]) => (
                  <div key={key}>
                    <h6>{key.replace('_', ' ').toUpperCase()}</h6>
                    <ul>
                      {strategies.map((strategy, idx) => <li key={idx}>{strategy}</li>)}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          )}

          {!futuristicMetrics && !futuristicLoading && (
            <div className="no-data">
              <p>Click "Refresh Metrics" to load futuristic inventory insights.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default InventoryManagement;
