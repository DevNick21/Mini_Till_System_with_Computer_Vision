import axios, { AxiosProgressEvent } from 'axios';
import { BetRecord, Customer, Alert } from '../types';

const API_BASE = process.env.REACT_APP_API_URL || '/api';

// Simplified API service
class ApiService {
  // Generic HTTP methods
  async get<T>(endpoint: string): Promise<T> {
    const response = await axios.get(`${API_BASE}${endpoint}`);
    return response.data;
  }

  async post<T>(endpoint: string, data?: any, config?: any): Promise<T> {
    const response = await axios.post(`${API_BASE}${endpoint}`, data, config);
    return response.data;
  }

  async put<T>(endpoint: string, data: any): Promise<T> {
    const response = await axios.put(`${API_BASE}${endpoint}`, data);
    return response.data;
  }

  async delete(endpoint: string): Promise<void> {
    await axios.delete(`${API_BASE}${endpoint}`);
  }

  // Bet operations
  async uploadBet(file: File, amount?: number, customerId?: number, onProgress?: (percent: number) => void): Promise<BetRecord> {
    const formData = new FormData();
    formData.append('file', file);
    if (typeof amount === 'number') formData.append('amount', String(amount));
    if (typeof customerId === 'number') formData.append('customerId', String(customerId));
    
    return this.post('/bet/upload', formData, {
      onUploadProgress: (evt: AxiosProgressEvent) => {
        if (onProgress && evt.total) {
          onProgress(Math.round((evt.loaded / evt.total) * 100));
        }
      }
    });
  }

  async getRecentBets(): Promise<BetRecord[]> {
    return this.get('/bet/recent');
  }

  // Customer operations
  async getCustomers(): Promise<Customer[]> {
    return this.get('/customers');
  }

  async createCustomer(name: string): Promise<Customer> {
    return this.post('/customers', { name });
  }

  // Alert operations
  async getAlerts(): Promise<Alert[]> {
    return this.get('/alerts');
  }

  async resolveAlert(id: number, resolvedBy?: string, notes?: string, customerId?: number): Promise<Alert> {
    return this.post(`/alerts/${id}/resolve`, {
      resolvedBy,
      notes,
      customerId
    });
  }
}

export const api = new ApiService();
