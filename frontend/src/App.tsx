import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Navbar, Nav, Container } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

import UploadComponent from './components/upload/UploadComponent';
import CustomersComponent from './components/customers/CustomersComponent';
import AlertsComponent from './components/alerts/AlertsComponent';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar bg="dark" variant="dark" expand="lg">
          <Container>
            <Navbar.Brand as={Link} to="/">BetFred Smart Customer Tracker</Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
              <Nav className="ms-auto">
                <Nav.Link as={Link} to="/">Upload Bet</Nav.Link>
                <Nav.Link as={Link} to="/customers">Customers</Nav.Link>
                <Nav.Link as={Link} to="/alerts">Alerts</Nav.Link>
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
