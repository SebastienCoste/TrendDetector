import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API_BASE = `${BACKEND_URL}/api/ui`;

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error(`API Error: ${error.response?.status} ${error.config?.url}`, error.response?.data);
        return Promise.reject(error);
      }
    );
  }

  // Model endpoints
  async getModels() {
    const response = await this.client.get('/models');
    return response.data;
  }

  // Vector generation endpoints
  async getAlgorithms() {
    const response = await this.client.get('/vector/algorithms');
    return response.data;
  }

  async generateVector(config) {
    const response = await this.client.post('/vector/generate', config);
    return response.data;
  }

  // Prediction endpoints
  async makePrediction(vector, modelType = 'classification') {
    const response = await this.client.post('/predict', {
      vector,
      model_type: modelType
    });
    return response.data;
  }

  // Drift testing endpoints
  async startDriftTest(config) {
    const response = await this.client.post('/drift-test/start', config);
    return response.data;
  }

  async getDriftTestStatus(testId) {
    const response = await this.client.get(`/drift-test/${testId}/status`);
    return response.data;
  }

  async getDriftTestResults(testId, limit = null) {
    let url = `/drift-test/${testId}/results`;
    if (limit) {
      url += `?limit=${limit}`;
    }
    const response = await this.client.get(url);
    return response.data;
  }

  async getActiveTests() {
    const response = await this.client.get('/drift-test/active');
    return response.data;
  }

  async cleanupDriftTest(testId) {
    const response = await this.client.delete(`/drift-test/${testId}`);
    return response.data;
  }
}

export const apiService = new ApiService();