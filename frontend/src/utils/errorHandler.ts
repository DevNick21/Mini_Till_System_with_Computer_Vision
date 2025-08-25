/**
 * Centralized error handling utilities
 */

export const handleApiError = (error: any, context: string): string => {
  console.error(`Error in ${context}:`, error);
  
  if (error?.response) {
    const status = error.response.status;
    const message = error.response.data || 'Unknown error';
    return `Request failed (${status}): ${message}`;
  }
  
  if (error?.message) {
    return error.message;
  }
  
  return 'An unexpected error occurred';
};

export const createErrorHandler = (context: string) => {
  return (error: any) => handleApiError(error, context);
};