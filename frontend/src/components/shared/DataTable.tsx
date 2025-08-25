import React, { useState } from 'react';
import { Table, Form, InputGroup, Card } from 'react-bootstrap';

export interface Column<T> {
  key: keyof T | string;
  header: string;
  render?: (value: any, item: T) => React.ReactNode;
  searchable?: boolean;
}

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  loading?: boolean;
  searchable?: boolean;
  title?: string;
}

function DataTable<T extends Record<string, any>>({
  data,
  columns,
  loading = false,
  searchable = true,
  title
}: DataTableProps<T>) {
  const [searchTerm, setSearchTerm] = useState('');

  const filteredData = searchable
    ? data.filter(item =>
        columns.some(col => {
          if (col.searchable === false) return false;
          const value = item[col.key as keyof T];
          return String(value).toLowerCase().includes(searchTerm.toLowerCase());
        })
      )
    : data;

  return (
    <div className="container mt-4">
      {title && <h2>{title}</h2>}
      
      <Card className="mb-4">
        <Card.Body>
          {searchable && (
            <InputGroup className="mb-3">
              <InputGroup.Text>
                🔍
              </InputGroup.Text>
              <Form.Control
                placeholder="Search..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </InputGroup>
          )}

          <Table responsive hover>
            <thead>
              <tr>
                {columns.map((col, index) => (
                  <th key={index}>{col.header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={columns.length} className="text-center">Loading...</td>
                </tr>
              ) : filteredData.length === 0 ? (
                <tr>
                  <td colSpan={columns.length} className="text-center">No data found</td>
                </tr>
              ) : (
                filteredData.map((item, index) => (
                  <tr key={item.id || index}>
                    {columns.map((col, colIndex) => {
                      const value = item[col.key as keyof T];
                      return (
                        <td key={`${item.id || index}-${colIndex}`}>
                          {col.render ? col.render(value, item) : String(value)}
                        </td>
                      );
                    })}
                  </tr>
                ))
              )}
            </tbody>
          </Table>
        </Card.Body>
      </Card>
    </div>
  );
}

export default DataTable;