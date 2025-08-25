import { useState, useCallback } from 'react';
import { handleApiError } from '../utils/errorHandler';

interface UseAsyncOperationResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  execute: () => Promise<void>;
  reset: () => void;
}

export function useAsyncOperation<T>(
  asyncFunction: () => Promise<T>,
  context: string
): UseAsyncOperationResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await asyncFunction();
      setData(result);
    } catch (err) {
      const errorMessage = handleApiError(err, context);
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [asyncFunction, context]);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, execute, reset };
}