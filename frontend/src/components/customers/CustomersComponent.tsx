import React, { useEffect, useCallback } from 'react';
import DataTable, { Column } from '../shared/DataTable';
import { api } from '../../services/api';
import { Customer } from '../../types';
import { useAsyncOperation } from '../../hooks/useAsyncOperation';
import LoadingState from '../shared/LoadingState';

const CustomersComponent: React.FC = () => {
  const getCustomers = useCallback(() => api.getCustomers(), []);
  
  const { data: customers, loading, error, execute } = useAsyncOperation(
    getCustomers,
    'loading customers'
  );

  useEffect(() => {
    execute();
  }, [execute]);

  const columns: Column<Customer>[] = [
    { key: 'id', header: 'ID' },
    { key: 'name', header: 'Name' }
  ];

  if (loading) return <LoadingState message="Loading customers..." />;
  if (error) return <div className="alert alert-danger">Error: {error}</div>;

  return (
    <DataTable
      data={customers || []}
      columns={columns}
      loading={false}
      title="Customers Management"
    />
  );
};

export default CustomersComponent;
