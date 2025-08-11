export interface BetRecord {
  id: number;
  amount: number;
  placedAt: string;
  outcome: string;
  writerClassification?: string | null;
  classificationConfidence?: number | null;
  customerId?: number | null;
  customerName?: string | null;
}

export interface Customer {
  id: number;
  name: string;
  email: string;
  phone: string;
  riskLevel: string;
}

export interface Alert {
  id: number;
  customerId?: number | null;
  message: string;
  alertType: string;
  severity: string;
  createdAt: string;
  isResolved: boolean;
}

export interface ThresholdRule {
  id: number;
  name: string;
  threshold: number;
  timeWindow: number;
  isActive: boolean;
}
