export interface BetRecord {
  id: number;
  customerId: number;
  customer: Customer;
  amount: number;
  odds: string;
  date: string;
  handwritingImageUrl: string;
  handwritingClassification: string;
  status: string;
}

export interface Customer {
  id: number;
  name: string;
  email: string;
  phone: string;
  isTagged: boolean;
  riskLevel: string;
}

export interface Alert {
  id: number;
  customerId: number;
  customer: Customer;
  message: string;
  type: string;
  date: string;
  isResolved: boolean;
}

export interface ThresholdRule {
  id: number;
  name: string;
  threshold: number;
  timeWindow: number;
  isActive: boolean;
}
