// frontend/src/services/api.js
import axios from 'axios';

class ApiService {
  static async getDashboardMetrics(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get('http://localhost:5000/api/dashboard-metrics', config);
      return response.data;
    } catch (error) {
      console.error('Error fetching dashboard metrics:', error);
      throw error;
    }
  }

  static async getDashboardData(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get('http://localhost:5000/api/dashboard/data', config);
      return response.data;
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      throw error;
    }
  }

  static async login(credentials) {
    try {
      const response = await axios.post('http://localhost:5000/api/login', credentials);
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
      const response = await axios.post('http://localhost:5000/api/register', userData);
      return response.data;
    } catch (error) {
      console.error('Error registering:', error);
      throw error;
    }
  }

  static async getProducts(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get('http://localhost:5000/api/products', config);
      return response.data;
    } catch (error) {
      console.error('Error fetching products:', error);
      throw error;
    }
  }

  static async getDemandForecast(productId, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get(`http://localhost:5000/api/demand-forecast/${productId}`, config);
      return response.data;
    } catch (error) {
      console.error('Error fetching demand forecast:', error);
      throw error;
    }
  }

  static async retrainDemandModel(productId, token, driftThreshold = 0.1) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post(`http://localhost:5000/api/retrain-demand-model/${productId}`, { drift_threshold: driftThreshold }, config);
      return response.data;
    } catch (error) {
      console.error('Error retraining demand model:', error);
      throw error;
    }
  }

  static async getInventoryAnalysis(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get('http://localhost:5000/api/inventory-analysis', config);
      return response.data;
    } catch (error) {
      console.error('Error fetching inventory analysis:', error);
      throw error;
    }
  }

  static async placeOrder(orderData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post('http://localhost:5000/api/place-order', orderData, config);
      return response.data;
    } catch (error) {
      console.error('Error placing order:', error);
      throw error;
    }
  }

  static async getLogisticsRoutes(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get('http://localhost:5000/api/logistics-routes', config);
      return response.data;
    } catch (error) {
      console.error('Error fetching logistics routes:', error);
      throw error;
    }
  }

  static async optimizeRoute(routeId, criteria, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post(`http://localhost:5000/api/optimize-route/${routeId}`, { optimization_criteria: criteria }, config);
      return response.data;
    } catch (error) {
      console.error('Error optimizing route:', error);
      throw error;
    }
  }

  static async implementOptimization(optimizationData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post('http://localhost:5000/api/implement-optimization', optimizationData, config);
      return response.data;
    } catch (error) {
      console.error('Error implementing optimization:', error);
      throw error;
    }
  }

  static async generateReport(reportType, reportData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post(`http://localhost:5000/api/reports/${reportType}`, reportData, config);
      return response.data;
    } catch (error) {
      console.error('Error generating report:', error);
      throw error;
    }
  }

  static async multiEchelonOptimize(optimizationData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post('http://localhost:5000/api/inventory/multi-echelon-optimize', optimizationData, config);
      return response.data;
    } catch (error) {
      console.error('Error performing multi-echelon optimization:', error);
      throw error;
    }
  }

  static async sendReorderAlert(alertData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post('http://localhost:5000/api/inventory/send-reorder-alert', alertData, config);
      return response.data;
    } catch (error) {
      console.error('Error sending reorder alert:', error);
      throw error;
    }
  }

  static async getEtaEstimates(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get('http://localhost:5000/api/eta-estimates', config);
      return response.data;
    } catch (error) {
      console.error('Error fetching ETA estimates:', error);
      throw error;
    }
  }

  static async getFuturisticMetrics(token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.get('http://localhost:5000/api/futuristic-inventory/metrics', config);
      return response.data;
    } catch (error) {
      console.error('Error fetching futuristic metrics:', error);
      throw error;
    }
  }

  static async optimizeSustainability(optimizationData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post('http://localhost:5000/api/futuristic-inventory/sustainability-optimize', optimizationData, config);
      return response.data;
    } catch (error) {
      console.error('Error optimizing sustainability:', error);
      throw error;
    }
  }

  static async assessSupplyChainRisks(riskData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post('http://localhost:5000/api/futuristic-inventory/risk-assessment', riskData, config);
      return response.data;
    } catch (error) {
      console.error('Error assessing supply chain risks:', error);
      throw error;
    }
  }

  static async predictDemandWithAI(predictionData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post('http://localhost:5000/api/futuristic-inventory/predict-demand', predictionData, config);
      return response.data;
    } catch (error) {
      console.error('Error predicting demand with AI:', error);
      throw error;
    }
  }

  static async integrateIoTData(iotData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post('http://localhost:5000/api/futuristic-inventory/iot-integration', iotData, config);
      return response.data;
    } catch (error) {
      console.error('Error integrating IoT data:', error);
      throw error;
    }
  }

  static async makeAutonomousDecisions(decisionData, token) {
    try {
      const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
      const response = await axios.post('http://localhost:5000/api/futuristic-inventory/autonomous-decisions', decisionData, config);
      return response.data;
    } catch (error) {
      console.error('Error making autonomous decisions:', error);
      throw error;
    }
  }
}

export default ApiService;