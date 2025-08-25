import React, { useEffect, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Navbar, Nav, Container, Badge } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

import UploadComponent from './components/upload/UploadComponent';
import CustomersComponent from './components/customers/CustomersComponent';
import AlertsComponent from './components/alerts/AlertsComponent';
import { api } from './services/api';
import { useAsyncOperation } from './hooks/useAsyncOperation';

function App() {
  const getAlerts = useCallback(() => api.getAlerts(), []);
  const { data: alerts, execute } = useAsyncOperation(getAlerts, 'loading alerts');
  
  useEffect(() => {
    // Load alerts once on mount
    execute();
  }, [execute]);

  const unresolvedCount = alerts?.filter(alert => !alert.isResolved).length || 0;

  return (
    <Router>
      <div className="App">
        <style>{`
          @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
          }
          
          .alert-badge {
            animation: pulse 2s infinite;
          }
        `}</style>
        
        <Navbar bg="dark" variant="dark" expand="lg">
          <Container>
            <Navbar.Brand as={Link} to="/">BetFred Smart Customer Tracker</Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
              <Nav className="ms-auto">
                <Nav.Link as={Link} to="/">Upload Bet</Nav.Link>
                <Nav.Link as={Link} to="/customers">Customers</Nav.Link>
                <Nav.Link as={Link} to="/alerts" className="position-relative">
                  Alerts
                  {unresolvedCount > 0 && (
                    <Badge 
                      bg="danger" 
                      className="position-absolute top-0 start-100 translate-middle alert-badge"
                      style={{ fontSize: '0.6em' }}
                    >
                      {unresolvedCount}
                    </Badge>
                  )}
                </Nav.Link>
              </Nav>
            </Navbar.Collapse>
          </Container>
        </Navbar>

        <Routes>
          <Route path="/" element={<UploadComponent />} />
          <Route path="/customers" element={<CustomersComponent />} />
          <Route path="/alerts" element={<AlertsComponent />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
