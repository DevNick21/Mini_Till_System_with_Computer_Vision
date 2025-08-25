import React from 'react';
import { Card, Row, Col } from 'react-bootstrap';

interface DashboardStatsProps {
  totalCustomers: number;
  totalBets: number;
  totalAlerts: number;
}

const DashboardStats: React.FC<DashboardStatsProps> = ({
  totalCustomers,
  totalBets,
  totalAlerts
}) => {
  return (
    <Row className="mb-4">
      <Col md={4}>
        <Card className="text-center bg-primary text-white">
          <Card.Body>
            <h2>{totalCustomers}</h2>
            <p>Total Customers</p>
          </Card.Body>
        </Card>
      </Col>
      <Col md={4}>
        <Card className="text-center bg-success text-white">
          <Card.Body>
            <h2>{totalBets}</h2>
            <p>Total Bets</p>
          </Card.Body>
        </Card>
      </Col>
      <Col md={4}>
        <Card className="text-center bg-warning text-dark">
          <Card.Body>
            <h2>{totalAlerts}</h2>
            <p>Total Alerts</p>
          </Card.Body>
        </Card>
      </Col>
    </Row>
  );
};

export default DashboardStats;