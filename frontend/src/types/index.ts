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
  resolvedAt?: string | null;
  resolvedBy?: string | null;
  resolutionNotes?: string | null;
  betRecordId?: number | null;
  customer?: any | null;
  betRecord?: any | null;
}

