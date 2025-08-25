import React, { useEffect, useCallback, useState } from 'react';
import { Card, Badge, Button, Alert, Form, Modal } from 'react-bootstrap';
import { api } from '../../services/api';
import { useAsyncOperation } from '../../hooks/useAsyncOperation';
import { formatDateTime } from '../../utils/dateFormatter';
import DashboardStats from '../shared/DashboardStats';
import LoadingState from '../shared/LoadingState';

const AlertsComponent: React.FC = () => {
  const getAlerts = useCallback(() => api.getAlerts(), []);
  
  const { data: alerts, loading, error, execute } = useAsyncOperation(
    getAlerts,
    'loading alerts'
  );

  const [showResolveModal, setShowResolveModal] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState<any>(null);
  const [customerName, setCustomerName] = useState('');
  const [resolving, setResolving] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  useEffect(() => {
    // Load alerts once on mount only
    handleRefresh();
  }, []);

  const handleRefresh = async () => {
    await execute();
    setLastUpdated(new Date());
  };

  // Check for unresolved alerts to trigger blinking
  const hasUnresolvedAlerts = alerts?.some(alert => !alert.isResolved) || false;

  // Simple mock stats based on alerts data
  const stats = {
    totalCustomers: 0,
    totalBets: 0,
    totalAlerts: alerts?.length || 0
  };

  const extractWriterIdFromMessage = (message: string): string | null => {
    const match = message.match(/Writer=(\w+)/);
    return match ? match[1] : null;
  };

  const handleResolveAlert = (alert: any) => {
    setSelectedAlert(alert);
    setCustomerName('');
    setShowResolveModal(true);
  };

  const handleResolveSubmit = async () => {
    if (!selectedAlert || !customerName.trim()) return;

    const alertId = selectedAlert.id;
    
    try {
      setResolving(true);
      
      // Optimistic update - immediately mark as resolved in local state
      if (alerts) {
        const updatedAlerts = alerts.map(alert => 
          alert.id === alertId 
            ? { ...alert, isResolved: true, resolvedBy: 'System Admin' }
            : alert
        );
        // This would need a way to update the local state, but for now we'll just refresh
      }
      
      // Create or find customer
      const customer = await api.createCustomer(customerName.trim());
      
      // Resolve alert with customer information
      await api.resolveAlert(
        alertId, 
        'System Admin', 
        `Linked Writer ${extractWriterIdFromMessage(selectedAlert.message)} to customer ${customerName}`,
        customer.id
      );
      
      // Close modal
      setShowResolveModal(false);
      setSelectedAlert(null);
      setCustomerName('');
      
      // Refresh to get server confirmation
      await handleRefresh();
      
    } catch (error) {
      console.error('Failed to resolve alert:', error);
      // Refresh to revert optimistic update if it failed
      await handleRefresh();
    } finally {
      setResolving(false);
    }
  };

  if (loading) return <LoadingState message="Loading alerts..." />;
  if (error) return <div className="alert alert-danger">Error: {error}</div>;

  return (
    <div className="container mt-4">
      <h2>Alerts & Dashboard</h2>
      
      <style>{`
        @keyframes blinkRed {
          0% { background-color: rgba(220, 53, 69, 0.1); }
          50% { background-color: rgba(220, 53, 69, 0.3); }
          100% { background-color: rgba(220, 53, 69, 0.1); }
        }
        
        .alert-blinking {
          animation: blinkRed 1.5s infinite;
        }
        
        .writer-highlight {
          background-color: #f8f9fa;
          padding: 8px;
          border-radius: 4px;
          border-left: 4px solid #dc3545;
          margin: 8px 0;
        }
      `}</style>
      
      <DashboardStats 
        totalCustomers={stats.totalCustomers}
        totalBets={stats.totalBets}
        totalAlerts={stats.totalAlerts}
      />
      
      <Card className={hasUnresolvedAlerts ? 'alert-blinking' : ''}>
        <Card.Header className="d-flex justify-content-between align-items-center">
          <div>
            <h4 className="mb-0">Recent Alerts</h4>
            {lastUpdated && (
              <small className="text-muted">
                Last updated: {lastUpdated.toLocaleTimeString()}
              </small>
            )}
          </div>
          <div className="d-flex align-items-center gap-2">
            {hasUnresolvedAlerts && (
              <Badge bg="danger" className="pulse">
                {alerts?.filter(a => !a.isResolved).length} Unresolved
              </Badge>
            )}
            <Button 
              variant="outline-primary" 
              size="sm" 
              onClick={handleRefresh}
              disabled={loading}
            >
              {loading ? 'Refreshing...' : 'Refresh'}
            </Button>
          </div>
        </Card.Header>
        <Card.Body>
          {!alerts || alerts.length === 0 ? (
            <Alert variant="info">No alerts found</Alert>
          ) : (
            alerts.map(alert => {
              const writerId = extractWriterIdFromMessage(alert.message);
              return (
                <Alert key={alert.id} variant={alert.isResolved ? 'success' : 'danger'} className="mb-3">
                  <div className="d-flex justify-content-between align-items-start">
                    <div className="flex-grow-1">
                      <div className="d-flex align-items-center mb-2">
                        <h5 className="mb-0">
                          Threshold Exceeded Alert
                          <Badge bg={alert.isResolved ? 'success' : 'danger'} className="ms-2">
                            {alert.isResolved ? 'Resolved' : 'URGENT'}
                          </Badge>
                        </h5>
                      </div>
                      
                      {writerId && (
                        <div className="writer-highlight">
                          <strong>Writer ID: {writerId}</strong>
                          <div className="text-muted small">
                            Multiple high-value bets detected from this handwriting pattern
                          </div>
                        </div>
                      )}
                      
                      <div className="mt-2">
                        <small className="text-muted">{alert.message}</small>
                      </div>
                      
                      <div className="mt-2">
                        <small>
                          <strong>Date:</strong> {formatDateTime(alert.createdAt)}
                        </small>
                        {alert.isResolved && alert.resolvedBy && (
                          <small className="ms-3">
                            <strong>Resolved by:</strong> {alert.resolvedBy}
                          </small>
                        )}
                      </div>
                    </div>
                    
                    {!alert.isResolved && (
                      <Button 
                        variant="danger"
                        size="sm"
                        onClick={() => handleResolveAlert(alert)}
                        className="ms-3"
                      >
                        Link Customer
                      </Button>
                    )}
                  </div>
                </Alert>
              );
            })
          )}
        </Card.Body>
      </Card>

      {/* Resolution Modal */}
      <Modal show={showResolveModal} onHide={() => setShowResolveModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Link Writer to Customer</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {selectedAlert && (
            <>
              <div className="alert alert-info">
                <strong>Writer ID:</strong> {extractWriterIdFromMessage(selectedAlert.message)}
                <br />
                <small>This will link all bets with this handwriting pattern to the customer</small>
              </div>
              
              <Form>
                <Form.Group className="mb-3">
                  <Form.Label>Customer Name</Form.Label>
                  <Form.Control
                    type="text"
                    placeholder="Enter customer name"
                    value={customerName}
                    onChange={(e) => setCustomerName(e.target.value)}
                    disabled={resolving}
                  />
                  <Form.Text className="text-muted">
                    A new customer will be created if this name doesn't exist
                  </Form.Text>
                </Form.Group>
              </Form>
            </>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowResolveModal(false)} disabled={resolving}>
            Cancel
          </Button>
          <Button 
            variant="primary" 
            onClick={handleResolveSubmit}
            disabled={resolving || !customerName.trim()}
          >
            {resolving ? 'Linking...' : 'Link Customer & Resolve'}
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
};

export default AlertsComponent;