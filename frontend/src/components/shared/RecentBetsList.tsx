import React from 'react';
import { Card, Button, Badge } from 'react-bootstrap';
import { BetRecord } from '../../types';
import { formatDate } from '../../utils/dateFormatter';
import LoadingState from './LoadingState';

interface RecentBetsListProps {
  bets: BetRecord[];
  loading: boolean;
  onRefresh: () => void;
}

const RecentBetsList: React.FC<RecentBetsListProps> = ({ bets, loading, onRefresh }) => {
  return (
    <Card>
      <Card.Header className="d-flex justify-content-between align-items-center">
        <span>Recent Bets</span>
        <Button variant="outline-secondary" size="sm" onClick={onRefresh} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </Button>
      </Card.Header>
      <Card.Body>
        {loading ? (
          <LoadingState message="Loading recent bets..." size="sm" />
        ) : bets.length === 0 ? (
          <p className="text-muted">No recent bets found</p>
        ) : (
          <div className="list-group">
            {bets.map(bet => (
              <div key={bet.id} className="list-group-item">
                <div className="d-flex w-100 justify-content-between">
                  <h5 className="mb-1">£{bet.amount.toFixed(2)}</h5>
                  <small>{formatDate(bet.placedAt)}</small>
                </div>
                <p className="mb-1">
                  <Badge bg={bet.writerClassification ? "success" : "secondary"}>
                    {bet.writerClassification 
                      ? `Writer ${bet.writerClassification} ${bet.classificationConfidence ? `(${(bet.classificationConfidence * 100).toFixed(1)}%)` : ''}` 
                      : 'Not classified'}
                  </Badge>
                </p>
                <small>CustomerId: {bet.customerId ?? 'Unknown'}</small>
              </div>
            ))}
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default RecentBetsList;