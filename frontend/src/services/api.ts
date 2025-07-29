import axios from 'axios';
import { BetRecord, Customer, Alert } from '../types';

const API_URL = process.env.REACT_APP_API_URL || 'https://localhost:5001/api';

export const apiService = {
  // Bet Records
  async uploadBetImage(file: File): Promise<string> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_URL}/bet/upload`, formData);
    return response.data;
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

  // Customers
  async getCustomers(): Promise<Customer[]> {
    const response = await axios.get(`${API_URL}/customers`);
    return response.data;
  },

  async tagCustomer(customerId: number, isTagged: boolean): Promise<Customer> {
    const response = await axios.put(`${API_URL}/customers/${customerId}/tag`, { isTagged });
    return response.data;
  },

  // Alerts
  async getAlerts(): Promise<Alert[]> {
    const response = await axios.get(`${API_URL}/alerts`);
    return response.data;
  },
  
  async resolveAlert(alertId: number): Promise<Alert> {
    const response = await axios.put(`${API_URL}/alerts/${alertId}/resolve`);
    return response.data;
  },

  // Dashboard
  async getDashboardStats(): Promise<any> {
    const response = await axios.get(`${API_URL}/dashboard/stats`);
    return response.data;
  }
};
