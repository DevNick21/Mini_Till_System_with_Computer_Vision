import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Form, Button, Card, Alert, Spinner, Toast, ToastContainer } from 'react-bootstrap';
import { api } from '../../services/api';
import { useToast } from '../../hooks/useToast';
import { useAsyncOperation } from '../../hooks/useAsyncOperation';
import { handleApiError } from '../../utils/errorHandler';
import RecentBetsList from '../shared/RecentBetsList';

const UploadComponent: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [amount, setAmount] = useState<number>(0);

  const { toasts, addToast, removeToast, addProgressToast, updateProgress, completeProgressToast } = useToast();
  
  const getRecentBets = useCallback(() => api.getRecentBets(), []);
  
  const { 
    data: recentBets, 
    loading: loadingRecent, 
    execute: loadRecentBets 
  } = useAsyncOperation(
    getRecentBets,
    'loading recent bets'
  );

  const lastFetchRef = useRef<number>(0);

  useEffect(() => {
    loadRecentBets();
  }, [loadRecentBets]);

  const throttledLoadRecentBets = async () => {
    const now = Date.now();
    if (now - lastFetchRef.current < 3000) {
      console.debug('[RecentBets] Throttled duplicate fetch');
      return;
    }
    lastFetchRef.current = now;
    await loadRecentBets();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);

      const reader = new FileReader();
      reader.onload = (e) => {
        if (e.target && e.target.result) {
          setPreview(e.target.result as string);
        }
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);

      const reader = new FileReader();
      reader.onload = (e) => {
        if (e.target && e.target.result) {
          setPreview(e.target.result as string);
        }
      };
      reader.readAsDataURL(droppedFile);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleAmountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setAmount(parseFloat(e.target.value) || 0);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please select an image file');
      return;
    }

    try {
      setUploading(true);
      setError(null);

      const progressToastId = addProgressToast('Uploading bet slip...');

      await api.uploadBet(file, amount, undefined, (progress: number) => {
        updateProgress(progressToastId, progress);
      });

      completeProgressToast(progressToastId, 'Bet uploaded successfully');
      addToast({ message: 'Bet uploaded (classification will appear shortly)', variant: 'success' });

      // Reset form
      setFile(null);
      setPreview(null);
      setAmount(0);

      // Refresh recent bets only
      await throttledLoadRecentBets();
      setTimeout(() => throttledLoadRecentBets(), 4000);

    } catch (err) {
      const errorMessage = handleApiError(err, 'uploading bet');
      setError(errorMessage);
      addToast({
        message: errorMessage,
        variant: 'danger',
        retry: () => {
          if (file && !uploading) {
            handleSubmit({ preventDefault: () => {} } as React.FormEvent);
          }
        }
      });
    } finally {
      setUploading(false);
    }
  };

  const clearForm = () => {
    setFile(null);
    setPreview(null);
    setError(null);
    setAmount(0);
  };

  return (
    <div className="container mt-4">
      <h2>Upload Bet Slip</h2>

      {error && (
        <Alert variant="danger" dismissible onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <div className="row">
        <div className="col-md-6">
          <Card className="mb-4">
            <Card.Body>
              <div
                className={`upload-area ${preview ? 'd-none' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
              >
                <p>Drag & Drop your bet slip image here or</p>
                <Button variant="outline-primary" onClick={() => document.getElementById('file-input')?.click()}>
                  Select File
                </Button>
                <input
                  type="file"
                  id="file-input"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="d-none"
                  aria-label="Select bet slip image file"
                />
              </div>

              {preview && (
                <div className="text-center">
                  <img 
                    src={preview} 
                    alt="Preview" 
                    className="img-fluid mb-3" 
                    style={{maxWidth: '300px', maxHeight: '300px'}} 
                  />
                  <div>
                    <Button variant="outline-danger" size="sm" onClick={clearForm} className="me-2">
                      Clear
                    </Button>
                  </div>
                </div>
              )}
            </Card.Body>
          </Card>

          {preview && (
            <Card>
              <Card.Body>
                <Form onSubmit={handleSubmit}>
                  <Form.Group className="mb-3">
                    <Form.Label>Amount</Form.Label>
                    <Form.Control
                      type="number"
                      name="amount"
                      value={amount}
                      onChange={handleAmountChange}
                      step="0.01"
                      min="0"
                      inputMode="decimal"
                      required
                    />
                  </Form.Group>

                  <div className="d-grid">
                    <Button type="submit" variant="primary" disabled={uploading}>
                      {uploading ? (
                        <>
                          <Spinner animation="border" size="sm" className="me-2" />
                          Processing...
                        </>
                      ) : (
                        'Process Bet'
                      )}
                    </Button>
                  </div>
                </Form>
              </Card.Body>
            </Card>
          )}
        </div>

        <div className="col-md-6">
          <RecentBetsList 
            bets={recentBets || []}
            loading={loadingRecent}
            onRefresh={throttledLoadRecentBets}
          />
        </div>
      </div>

      <ToastContainer position="bottom-end" className="p-3">
        {toasts.map(t => (
          <Toast
            key={t.id}
            bg={t.variant === 'info-progress' ? 'info' : t.variant}
            onClose={() => removeToast(t.id)}
            delay={t.variant === 'success' ? 3500 : 5000}
            autohide={t.variant !== 'danger' && t.variant !== 'info-progress'}
          >
            <Toast.Header closeButton className="d-flex justify-content-between w-100">
              <strong className="me-auto">Upload</strong>
              <small>{new Date().toLocaleTimeString()}</small>
            </Toast.Header>
            <Toast.Body className={t.variant === 'success' ? 'text-white' : ''}>
              <div className="d-flex justify-content-between align-items-start gap-2">
                <span className="flex-grow-1">
                  {t.message}
                  {t.variant === 'info-progress' && (
                    <div className="mt-2">
                      <div className="progress">
                        <div 
                          className="progress-bar progress-bar-striped progress-bar-animated" 
                          role="progressbar" 
                          style={{width: `${t.progress || 0}%`}}
                        >
                          {t.progress || 0}%
                        </div>
                      </div>
                    </div>
                  )}
                </span>
                {t.retry && !t.inProgress && (
                  <Button size="sm" variant="light" onClick={() => { t.retry && t.retry(); removeToast(t.id); }}>
                    Retry
                  </Button>
                )}
              </div>
            </Toast.Body>
          </Toast>
        ))}
      </ToastContainer>
    </div>
  );
};

export default UploadComponent;