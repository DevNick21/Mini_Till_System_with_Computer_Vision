import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders BetFred app title', () => {
  render(<App />);
  const titleElement = screen.getByText(/BetFred Smart Customer Tracker/i);
  expect(titleElement).toBeInTheDocument();
});
