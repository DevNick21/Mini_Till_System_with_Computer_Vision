import React, { useState, useEffect } from 'react';
import { Card, Badge, Button, Alert, Row, Col } from 'react-bootstrap';
import { apiService } from '../../services/api';
import { Alert as AlertType } from '../../types';

const AlertsComponent: React.FC = () => {
  const [alerts, setAlerts] = useState<AlertType[]>([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState<any>({
    totalCustomers: 0,
    totalBets: 0,
    totalAlerts: 0
  });

  useEffect(() => {
    loadAlerts();
    loadDashboardStats();
  }, []);

  const loadAlerts = async () => {
    try {
      setLoading(true);
      const data = await apiService.getAlerts();
      setAlerts(data);
      setLoading(false);
    } catch (error) {
      console.error('Error loading alerts:', error);
      setLoading(false);
    }
  };

  const loadDashboardStats = async () => {
    try {
      const data = await apiService.getDashboardStats();
      setStats(data);
    } catch (error) {
      console.error('Error loading dashboard stats:', error);
    }
  };

  const handleResolveAlert = (alertId: number) => {
    // Optimistically update UI; backend resolve endpoint not implemented
    setAlerts(prev => prev.map(alert =>
      alert.id === alertId ? { ...alert, isResolved: true } : alert
    ));
    loadDashboardStats();
  };

  return (
    <div className="container mt-4">
      <h2>Alerts & Dashboard</h2>
      
      <Row className="mb-4">
        <Col md={4}>
          <Card className="text-center bg-primary text-white">
            <Card.Body>
              <h2>{stats.totalCustomers}</h2>
              <p>Total Customers</p>
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="text-center bg-success text-white">
            <Card.Body>
              <h2>{stats.totalBets}</h2>
              <p>Total Bets</p>
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="text-center bg-warning text-dark">
            <Card.Body>
              <h2>{stats.totalAlerts}</h2>
              <p>Total Alerts</p>
            </Card.Body>
          </Card>
        </Col>
      </Row>
      
      <Card>
        <Card.Header>
          <h4>Recent Alerts</h4>
        </Card.Header>
        <Card.Body>
          {loading ? (
            <p className="text-center">Loading alerts...</p>
          ) : alerts.length === 0 ? (
            <Alert variant="info">No alerts found</Alert>
          ) : (
            alerts.map(alert => (
        <Alert key={alert.id} variant={alert.isResolved ? 'success' : 'warning'} className="mb-3">
                <div className="d-flex justify-content-between align-items-center">
                  <div>
                    <h5>
                      {alert.message}
          <Badge bg={alert.isResolved ? 'success' : 'secondary'} className="ms-2">
                        {alert.isResolved ? 'Resolved' : 'Pending'}
                      </Badge>
                    </h5>
                    <div>
          <strong>CustomerId:</strong> {alert.customerId ?? 'N/A'}
                    </div>
                    <div>
                      <strong>Date:</strong> {new Date(alert.createdAt).toLocaleString()}
                    </div>
                  </div>
                  
                  {!alert.isResolved && (
                    <Button 
                      variant="outline-dark"
                      size="sm"
                      onClick={() => handleResolveAlert(alert.id)}
                    >
                      Mark as Resolved
                    </Button>
                  )}
                </div>
              </Alert>
            ))
          )}
        </Card.Body>
      </Card>
    </div>
  );
};

export default AlertsComponent;
