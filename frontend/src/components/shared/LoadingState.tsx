import React from 'react';

interface LoadingStateProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
}

const LoadingState: React.FC<LoadingStateProps> = ({ 
  message = 'Loading...', 
  size = 'md' 
}) => {
  return (
    <div className={`text-center ${size === 'lg' ? 'p-4' : 'p-2'}`}>
      <p className="text-muted">{message}</p>
    </div>
  );
};

export default LoadingState;