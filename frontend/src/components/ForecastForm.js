import React, { useState } from 'react';
import './ForecastForm.css';

function ForecastForm({ token }) {
  const [formData, setFormData] = useState({
    product: '',
    forecast_horizon: 30,
    sales_data_path: 'data/sales.csv',
    weather_data_path: ''
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleTrainModels = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/forecast/train', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sales_data_path: formData.sales_data_path,
          weather_data_path: formData.weather_data_path || null
        }),
      });

      const data = await response.json();

      if (response.ok) {
        alert('Models trained successfully!');
      } else {
        setError(data.error || 'Training failed');
      }
    } catch (err) {
      setError('Network error');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateForecast = async () => {
    if (!formData.product) {
      setError('Please enter a product name');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/forecast/predict', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          product: formData.product,
          forecast_horizon: parseInt(formData.forecast_horizon)
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setResults(data);
      } else {
        setError(data.error || 'Forecast generation failed');
      }
    } catch (err) {
      setError('Network error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="forecast-form">
      <h2>Demand Forecasting</h2>

      <div className="form-section">
        <h3>Model Training</h3>
        <div className="form-group">
          <label>Sales Data Path:</label>
          <input
            type="text"
            name="sales_data_path"
            value={formData.sales_data_path}
            onChange={handleChange}
            placeholder="data/sales.csv"
          />
        </div>

        <div className="form-group">
          <label>Weather Data Path (Optional):</label>
          <input
            type="text"
            name="weather_data_path"
            value={formData.weather_data_path}
            onChange={handleChange}
            placeholder="data/weather.csv"
          />
        </div>

        <button
          onClick={handleTrainModels}
          disabled={loading}
          className="train-button"
        >
          {loading ? 'Training...' : 'Train Models'}
        </button>
      </div>

      <div className="form-section">
        <h3>Generate Forecast</h3>
        <div className="form-group">
          <label>Product Name:</label>
          <input
            type="text"
            name="product"
            value={formData.product}
            onChange={handleChange}
            placeholder="Enter product name"
            required
          />
        </div>

        <div className="form-group">
          <label>Forecast Horizon (days):</label>
          <input
            type="number"
            name="forecast_horizon"
            value={formData.forecast_horizon}
            onChange={handleChange}
            min="1"
            max="365"
          />
        </div>

        <button
          onClick={handleGenerateForecast}
          disabled={loading}
          className="forecast-button"
        >
          {loading ? 'Generating...' : 'Generate Forecast'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {results && (
        <div className="results-section">
          <h3>Forecast Results</h3>

          {results.summary && (
            <div className="forecast-summary">
              <h4>Summary for {formData.product}</h4>
              <div className="summary-grid">
                <div className="summary-item">
                  <span className="summary-label">Total Predicted Demand:</span>
                  <span className="summary-value">{results.summary.total_predicted_demand}</span>
                </div>
                <div className="summary-item">
                  <span className="summary-label">Average Daily Demand:</span>
                  <span className="summary-value">{results.summary.average_daily_demand.toFixed(1)}</span>
                </div>
                <div className="summary-item">
                  <span className="summary-label">Max Daily Demand:</span>
                  <span className="summary-value">{results.summary.max_daily_demand}</span>
                </div>
                <div className="summary-item">
                  <span className="summary-label">Demand Volatility:</span>
                  <span className="summary-value">{results.summary.demand_volatility.toFixed(2)}</span>
                </div>
              </div>
            </div>
          )}

          {results.forecast && (
            <div className="forecast-table">
              <h4>Daily Forecast</h4>
              <table>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Predicted Demand</th>
                    <th>Confidence</th>
                    <th>Upper Bound</th>
                    <th>Lower Bound</th>
                  </tr>
                </thead>
                <tbody>
                  {results.forecast.slice(0, 10).map((item, index) => (
                    <tr key={index}>
                      <td>{item.date}</td>
                      <td>{item.predicted_demand}</td>
                      <td>{(item.confidence_score * 100).toFixed(1)}%</td>
                      <td>{item.upper_bound}</td>
                      <td>{item.lower_bound}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {results.forecast.length > 10 && (
                <p className="table-note">Showing first 10 days. Total forecast: {results.forecast.length} days.</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ForecastForm;
