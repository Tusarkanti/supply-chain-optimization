import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';
import ApiService from '../api';
import './DemandForecasting.css';

function DemandForecasting({ token }) {
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState('');
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchProducts();
  }, [token]);

  const fetchProducts = async () => {
    try {
      const data = await ApiService.getProducts(token);
      setProducts(data.products || []);
    } catch (err) {
      setError(err.message || 'Failed to fetch products');
    }
  };

  const handleForecast = async () => {
    if (!selectedProduct) return;

    setLoading(true);
    setError('');
    setForecast(null);

    try {
      const data = await ApiService.getDemandForecast(selectedProduct, token);
      setForecast(data);
    } catch (err) {
      setError(err.message || 'Failed to generate forecast');
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async () => {
    if (!selectedProduct) return;

    setRetraining(true);
    setError('');

    try {
      const data = await ApiService.retrainDemandModel(selectedProduct, token);
      if (data.success) {
        alert(`Model retraining ${data.result.retrained ? 'completed successfully' : 'not needed'}: ${data.result.status}`);
        // Optionally refresh the forecast after retraining
        if (data.result.retrained) {
          setForecast(null); // Clear current forecast to encourage re-generation
        }
      } else {
        setError('Failed to retrain model');
      }
    } catch (err) {
      setError(err.message || 'Failed to retrain model');
    } finally {
      setRetraining(false);
    }
  };

  return (
    <div className="demand-forecasting">
      <div className="forecast-header">
        <h2>Demand Forecasting</h2>
        <p>Predict future demand for your products using advanced ML models</p>
      </div>

      <div className="forecast-controls">
        <div className="control-group">
          <label htmlFor="product-select">Select Product:</label>
          <select
            id="product-select"
            value={selectedProduct}
            onChange={(e) => setSelectedProduct(e.target.value)}
          >
            <option value="">Choose a product...</option>
            {products.map((product) => (
              <option key={product.id} value={product.id}>
                {product.name}
              </option>
            ))}
          </select>
        </div>

        <button
          className="forecast-button"
          onClick={handleForecast}
          disabled={!selectedProduct || loading}
        >
          {loading ? 'Generating Forecast...' : 'Generate Forecast'}
        </button>

        <button
          className="retrain-button"
          onClick={handleRetrain}
          disabled={!selectedProduct || retraining}
        >
          {retraining ? 'Retraining Model...' : 'Retrain Model'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {loading && (
        <div className="forecast-results">
          <h3>Generating Forecast...</h3>
          <div className="skeleton-loader">
            <div className="skeleton-summary">
              <div className="skeleton-card"></div>
              <div className="skeleton-card"></div>
              <div className="skeleton-card"></div>
            </div>
            <div className="skeleton-ensemble">
              <div className="skeleton-line"></div>
              <div className="skeleton-line"></div>
              <div className="skeleton-line"></div>
            </div>
            <div className="skeleton-table">
              <div className="skeleton-header"></div>
              {[...Array(5)].map((_, i) => (
                <div key={i} className="skeleton-row">
                  <div className="skeleton-cell"></div>
                  <div className="skeleton-cell"></div>
                  <div className="skeleton-cell"></div>
                  <div className="skeleton-cell"></div>
                  <div className="skeleton-cell wide"></div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {forecast && (
        <div className="forecast-results">
          <h3>Forecast Results for {products.find(p => p.id === selectedProduct)?.name || selectedProduct}</h3>

          <div className="forecast-summary">
            <div className="summary-card">
              <h4>Total Predicted Demand</h4>
              <p className="summary-value">{forecast.total_predicted_demand ?? '-'}</p>
            </div>
            <div className="summary-card">
              <h4>Average Daily Demand</h4>
              <p className="summary-value">{forecast.average_daily_demand?.toFixed(2) ?? '-'}</p>
            </div>
            <div className="summary-card">
              <h4>Demand Volatility</h4>
              <p className="summary-value">{forecast.demand_volatility?.toFixed(2) ?? '-'}</p>
            </div>
          </div>

          {/* Ensemble Model Details */}
          {forecast.ensemble_weights && (
            <div className="ensemble-details">
              <h4>Ensemble Model Weights</h4>
              <div className="weights-list">
                {Object.entries(forecast.ensemble_weights).map(([model, weight]) => (
                  <div key={model} className="weight-item">
                    <span className="model-name">{model}:</span>
                    <span className="weight-value">{(weight * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
              <p className="weights-note">Weights based on cross-validation performance (higher weight = better performing model)</p>
            </div>
          )}

          {/* Model Performance Comparison */}
          {forecast.ensemble_weights && (
            <div className="performance-comparison">
              <h4>Model Performance Comparison</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={Object.entries(forecast.ensemble_weights).map(([model, weight]) => ({ model, performance: weight * 100 }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="performance" fill="#8884d8" name="Performance Score (%)" />
                </BarChart>
              </ResponsiveContainer>
              <p className="performance-note">Performance scores derived from ensemble weights (higher = better cross-validation performance)</p>
            </div>
          )}

          {/* Model Performance Comparison */}
          {forecast.forecast_details && forecast.forecast_details.length > 0 && (
            <div className="model-performance">
              <h4>Model Performance Comparison</h4>
              <div className="performance-chart">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={forecast.forecast_details.slice(0, 14)}> {/* Show first 14 days for clarity */}
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="predicted_demand"
                      stroke="#8884d8"
                      strokeWidth={3}
                      name="Ensemble Prediction"
                    />
                    {forecast.forecast_details[0]?.individual_predictions &&
                      Object.keys(forecast.forecast_details[0].individual_predictions).map((model, index) => (
                        <Line
                          key={model}
                          type="monotone"
                          dataKey={`individual_predictions.${model}`}
                          stroke={`hsl(${index * 60}, 70%, 50%)`}
                          strokeWidth={1}
                          strokeDasharray="5 5"
                          name={model}
                          dot={false}
                        />
                      ))
                    }
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="chart-note">Solid line: Ensemble prediction | Dashed lines: Individual model predictions</p>
            </div>
          )}

          <div className="forecast-details">
            <h4>Detailed Forecast</h4>
            {forecast.forecast_details?.length > 0 ? (
              <div className="forecast-table">
                <div className="table-header">
                  <span>Date</span>
                  <span>Ensemble Prediction</span>
                  <span>Confidence</span>
                  <span>Range</span>
                  <span>Individual Models</span>
                </div>
                {forecast.forecast_details.map((item, index) => (
                  <div key={index} className="table-row">
                    <span>{item.date}</span>
                    <span>{item.predicted_demand ?? '-'}</span>
                    <span>{item.confidence_score != null ? (item.confidence_score * 100).toFixed(1) + '%' : '-'}</span>
                    <span>{item.lower_bound ?? '-'} - {item.upper_bound ?? '-'}</span>
                    <div className="individual-preds">
                      {item.individual_predictions ? Object.entries(item.individual_predictions).map(([model, pred]) => (
                        <div key={model} className="pred-item">
                          <small>{model}: {Math.round(pred)}</small>
                        </div>
                      )) : <small>-</small>}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p>No forecast details available</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default DemandForecasting;
