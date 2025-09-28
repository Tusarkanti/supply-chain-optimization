import axios from 'axios';

// Use environment variable for API URL, fallback to production backend
const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://supply-chain-optimization-2.onrender.com';

class ApiService {
  static async getDashboardMetrics(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get(`${API_BASE_URL}/api/dashboard-metrics`, config);
      return response.data;
    } catch (error) {
      console.error('Error fetching dashboard metrics:', error);
      throw error;
    }
  }

  static async getDashboardData(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get(`${API_BASE_URL}/api/dashboard/data`, config);
      return response.data;
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      throw error;
    }
  }

  static async login(credentials) {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/login`, credentials);
      return response.data;
    } catch (error) {
      console.error('Error logging in:', error);
      if (error.response && error.response.data) {
        throw error.response.data;
      }
      throw error;
    }
  }

  static async register(userData) {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/register`, userData);
      return response.data;
    } catch (error) {
      console.error('Error registering:', error);
      throw error;
    }
  }

  static async getProducts(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get(`${API_BASE_URL}/api/products`, config);
      return response.data;
    } catch (error) {
      console.error('Error fetching products:', error);
      throw error;
    }
  }

  static async getDemandForecast(productId, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get(`${API_BASE_URL}/api/demand-forecast/${productId}`, config);
      return response.data;
    } catch (error) {
      console.error('Error fetching demand forecast:', error);
      throw error;
    }
  }

  static async retrainDemandModel(productId, token, driftThreshold = 0.1) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get(`${API_BASE_URL}/api/futuristic-inventory/metrics`, config);
      return response.data;
    } catch (error) {
      console.error('Error fetching futuristic metrics:', error);
      throw error;
    }
  }

  static async optimizeSustainability(optimizationData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post(`${API_BASE_URL}/api/futuristic-inventory/sustainability-optimize`, optimizationData, config);
      return response.data;
    } catch (error) {
      console.error('Error optimizing sustainability:', error);
      throw error;
    }
  }

  static async assessSupplyChainRisks(riskData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post(`${API_BASE_URL}/api/futuristic-inventory/risk-assessment`, riskData, config);
      return response.data;
    } catch (error) {
      console.error('Error assessing supply chain risks:', error);
      throw error;
    }
  }

  static async predictDemandWithAI(predictionData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post(`${API_BASE_URL}/api/futuristic-inventory/predict-demand`, predictionData, config);
      return response.data;
    } catch (error) {
      console.error('Error predicting demand with AI:', error);
      throw error;
    }
  }

  static async integrateIoTData(iotData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post(`${API_BASE_URL}/api/futuristic-inventory/iot-integration`, iotData, config);
      return response.data;
    } catch (error) {
      console.error('Error integrating IoT data:', error);
      throw error;
    }
  }

  static async makeAutonomousDecisions(decisionData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post(`${API_BASE_URL}/api/futuristic-inventory/autonomous-decisions`, decisionData, config);
      return response.data;
    } catch (error) {
      console.error('Error making autonomous decisions:', error);
      throw error;
    }
  }
}

export default ApiService;
