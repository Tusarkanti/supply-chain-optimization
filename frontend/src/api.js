import axios from 'axios';

// Use environment variable for API URL, fallback to production backend
//const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://supply-chain-optimization-v93t.onrender.com';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';


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

  static async getEtaEstimates(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get(`${API_BASE_URL}/api/eta-estimates`, config);
      return response.data;
    } catch (error) {
      console.error('Error fetching ETA estimates:', error);
      throw error;
    }
  }

  static async login(credentials) {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/login`, credentials, {
        headers: { 'Content-Type': 'application/json' }
      });
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
      const response = await axios.post(`${API_BASE_URL}/api/register`, userData, {
        headers: { 'Content-Type': 'application/json' }
      });
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
      const parsedProductId = parseInt(productId, 10);
      if (isNaN(parsedProductId)) {
        throw new Error('Invalid product ID');
      }
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get(`${API_BASE_URL}/api/demand-forecast/${parsedProductId}`, config);
      return response.data;
    } catch (error) {
      console.error('Error fetching demand forecast:', error);
      throw error;
    }
  }

  static async retrainDemandModel(productId, token, driftThreshold = 0.1) {
    try {
      const parsedProductId = parseInt(productId, 10);
      if (isNaN(parsedProductId)) {
        throw new Error('Invalid product ID');
      }
      const config = token ? { headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } } : { headers: { 'Content-Type': 'application/json' } };
      const response = await axios.post(`${API_BASE_URL}/api/retrain-demand-model/${parsedProductId}`, { drift_threshold: driftThreshold }, config);
      return response.data;
    } catch (error) {
      console.error('Error retraining demand model:', error);
      throw error;
    }
  }

  static async getInventoryAnalysis(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get(`${API_BASE_URL}/api/inventory-analysis`, config);
      return response.data;
    } catch (error) {
      console.error('Error fetching inventory analysis:', error);
      throw error;
    }
  }

  static async getFuturisticMetrics(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get(`${API_BASE_URL}/api/futuristic-inventory/metrics`, config);
      return response.data;
    } catch (error) {
      console.error('Error fetching futuristic metrics:', error);
      throw error;
    }
  }

  static async getLogisticsRoutes(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get(`${API_BASE_URL}/api/logistics-routes`, config);
      return response.data;
    } catch (error) {
      console.error('Error fetching logistics routes:', error);
      throw error;
    }
  }

  static async optimizeRoute(routeId, criteria, token) {
    try {
      const parsedRouteId = parseInt(routeId, 10);
      if (isNaN(parsedRouteId)) {
        throw new Error('Invalid route ID');
      }
      const config = token ? { headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } } : { headers: { 'Content-Type': 'application/json' } };
      const response = await axios.post(`${API_BASE_URL}/api/optimize-route/${parsedRouteId}`, { optimization_criteria: criteria }, config);
      return response.data;
    } catch (error) {
      console.error('Error optimizing route:', error);
      throw error;
    }
  }

  static async implementOptimization(data, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } } : { headers: { 'Content-Type': 'application/json' } };
      const response = await axios.post(`${API_BASE_URL}/api/implement-optimization`, data, config);
      return response.data;
    } catch (error) {
      console.error('Error implementing optimization:', error);
      throw error;
    }
  }

  static async optimizeSustainability(optimizationData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } } : { headers: { 'Content-Type': 'application/json' } };
      const response = await axios.post(`${API_BASE_URL}/api/futuristic-inventory/sustainability-optimize`, optimizationData, config);
      return response.data;
    } catch (error) {
      console.error('Error optimizing sustainability:', error);
      throw error;
    }
  }

  static async assessSupplyChainRisks(riskData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } } : { headers: { 'Content-Type': 'application/json' } };
      const response = await axios.post(`${API_BASE_URL}/api/futuristic-inventory/risk-assessment`, riskData, config);
      return response.data;
    } catch (error) {
      console.error('Error assessing supply chain risks:', error);
      throw error;
    }
  }

  static async predictDemandWithAI(predictionData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } } : { headers: { 'Content-Type': 'application/json' } };
      const response = await axios.post(`${API_BASE_URL}/api/futuristic-inventory/predict-demand`, predictionData, config);
      return response.data;
    } catch (error) {
      console.error('Error predicting demand with AI:', error);
      throw error;
    }
  }

  static async integrateIoTData(iotData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } } : { headers: { 'Content-Type': 'application/json' } };
      const response = await axios.post(`${API_BASE_URL}/api/futuristic-inventory/iot-integration`, iotData, config);
      return response.data;
    } catch (error) {
      console.error('Error integrating IoT data:', error);
      throw error;
    }
  }

  static async makeAutonomousDecisions(decisionData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } } : { headers: { 'Content-Type': 'application/json' } };
      const response = await axios.post(`${API_BASE_URL}/api/futuristic-inventory/autonomous-decisions`, decisionData, config);
      return response.data;
    } catch (error) {
      console.error('Error making autonomous decisions:', error);
      throw error;
    }
  }
}

export default ApiService;
