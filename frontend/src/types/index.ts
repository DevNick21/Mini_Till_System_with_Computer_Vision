export interface BetRecord {
  id: number;
  amount: number;
  placedAt: string;
  writerClassification?: string | null;
  classificationConfidence?: number | null;
  customerId?: number | null;
}

export interface Customer {
  id: number;
  name: string;
}

export interface Alert {
  id: number;
  customerId?: number | null;
  message: string;
  alertType: string;
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
