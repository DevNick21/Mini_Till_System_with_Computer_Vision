/**
 * Date formatting utilities
 */

export const formatDate = (date: string): string => {
  return new Date(date).toLocaleDateString();
};

export const formatDateTime = (date: string): string => {
  return new Date(date).toLocaleString();
};

export const formatTime = (date: string): string => {
  return new Date(date).toLocaleTimeString();
};