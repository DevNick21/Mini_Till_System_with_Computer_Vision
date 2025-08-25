import { useState, useCallback } from 'react';

interface ToastItem {
  id: number;
  message: string;
  variant: 'success' | 'danger' | 'info' | 'info-progress';
  progress?: number;
  inProgress?: boolean;
  retry?: () => void;
}

export function useToast() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const addToast = useCallback((toast: Omit<ToastItem, 'id'>) => {
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, ...toast }]);
    return id;
  }, []);

  const removeToast = useCallback((id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const updateToast = useCallback((id: number, updates: Partial<ToastItem>) => {
    setToasts(prev => prev.map(t => t.id === id ? { ...t, ...updates } : t));
  }, []);

  const addProgressToast = useCallback((message: string): number => {
    return addToast({
      message,
      variant: 'info-progress',
      progress: 0,
      inProgress: true
    });
  }, [addToast]);

  const updateProgress = useCallback((id: number, progress: number) => {
    updateToast(id, { progress });
  }, [updateToast]);

  const completeProgressToast = useCallback((id: number, successMessage: string) => {
    updateToast(id, { 
      progress: 100, 
      inProgress: false, 
      message: successMessage 
    });
    
    // Remove after showing completion
    setTimeout(() => removeToast(id), 1000);
  }, [updateToast, removeToast]);

  return {
    toasts,
    addToast,
    removeToast,
    updateToast,
    addProgressToast,
    updateProgress,
    completeProgressToast
  };
}