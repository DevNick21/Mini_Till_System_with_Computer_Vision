import axios, { AxiosProgressEvent } from 'axios';
import { BetRecord, Customer, Alert } from '../types';

const API_URL = process.env.REACT_APP_API_URL || '/api';

export const apiService = {
  // Bet Records
  async uploadBetImage(file: File, amount?: number, customerId?: number, onProgress?: (percent: number) => void): Promise<string> {
    const formData = new FormData();
    formData.append('file', file);
    if (typeof amount === 'number') formData.append('amount', String(amount));
    if (typeof customerId === 'number') formData.append('customerId', String(customerId));
    const response = await axios.post(`${API_URL}/bet/upload`, formData, {
      onUploadProgress: (evt: AxiosProgressEvent) => {
        if (onProgress && evt.total) {
          const pct = Math.round((evt.loaded / evt.total) * 100);
          onProgress(pct);
        }
      }
    });
    // Backend may return the whole object or just the id string; normalize to id string
    const data = response.data as any;
    return typeof data === 'string' ? data : String(data?.id ?? '');
  },

  async createBet(betData: Partial<BetRecord>): Promise<BetRecord> {
    const response = await axios.post(`${API_URL}/bet`, betData);
    return response.data;
  },
  
  async updateBet(id: number, betData: Partial<BetRecord>): Promise<BetRecord> {
    const response = await axios.put(`${API_URL}/bet/${id}`, betData);
    return response.data;
  },

  async getRecentBets(): Promise<BetRecord[]> {
    const response = await axios.get(`${API_URL}/bet/recent`);
    return response.data;
  },

  // SSE removed; rely on polling for now

  // Customers
  async getCustomers(): Promise<Customer[]> {
    const response = await axios.get(`${API_URL}/customers`);
    return response.data;
  },


  // Alerts
  async getAlerts(): Promise<Alert[]> {
    const response = await axios.get(`${API_URL}/alerts`);
    return response.data;
  },

  // Dashboard
  async getDashboardStats(): Promise<any> {
  // Backend provides dashboard stats via Alerts or System controller
  const response = await axios.get(`${API_URL}/alerts/dashboard`);
    return response.data;
  }
};
