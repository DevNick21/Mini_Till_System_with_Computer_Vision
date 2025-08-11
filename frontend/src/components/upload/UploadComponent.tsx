import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Form, Button, Card, Alert, Spinner, Toast, ToastContainer, Collapse, ProgressBar } from 'react-bootstrap';
import { apiService } from '../../services/api';
import { BetRecord } from '../../types';

const UploadComponent: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [errorDetails, setErrorDetails] = useState<string | null>(null);
  const [betData, setBetData] = useState<Partial<BetRecord>>({
    amount: 0
  });
  const [recentBets, setRecentBets] = useState<BetRecord[]>([]);
  const [loadingRecent, setLoadingRecent] = useState(false);
  interface ToastItem { id: number; message: string; variant: 'success' | 'danger' | 'info' | 'info-progress'; retry?: () => void; progress?: number; inProgress?: boolean; }
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const pushToast = useCallback((t: Omit<ToastItem, 'id'>) => {
    setToasts(prev => [...prev, { id: Date.now() + Math.random(), ...t }]);
  }, []);
  const removeToast = (id: number) => setToasts(prev => prev.filter(t => t.id !== id));
  const [showErrorDetails, setShowErrorDetails] = useState(false);

  useEffect(() => {
    loadRecentBets();
    const es = apiService.openClassificationUpdates(update => {
      setRecentBets(prev => prev.map(b => b.id === update.betId ? { ...b, writerClassification: update.writerClassification, classificationConfidence: update.confidence } : b));
    });
    return () => { es.close(); };
  }, []);

  const lastFetchRef = useRef<number>(0);
  const loadRecentBets = async () => {
    const now = Date.now();
    if (now - lastFetchRef.current < 3000) { // throttle to 1 call per 3s
      console.debug('[RecentBets] Throttled duplicate fetch');
      return;
    }
    lastFetchRef.current = now;
    try {
      console.debug('[RecentBets] Fetching recent bets');
      setLoadingRecent(true);
      const bets = await apiService.getRecentBets();
      setRecentBets(bets);
    } catch (err) {
      console.error('Error loading recent bets:', err);
    } finally {
      setLoadingRecent(false);
    }
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
  // removed uploadSuccess state
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
  // removed uploadSuccess state
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setBetData(prev => ({
      ...prev,
      [name]: name === 'amount' ? parseFloat(value) : value
    }));
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
      setErrorDetails(null);
      // Upload image along with stake (amount); backend will create the bet and return the created bet (or its id)
      // Create an in-progress toast
      let progressToastId = Date.now() + Math.random();
      setToasts(prev => [...prev, { id: progressToastId, message: 'Uploading bet slip...', variant: 'info-progress', progress: 0, inProgress: true }]);

  await apiService.uploadBetImage(file, betData.amount, undefined, (pct: number) => {
        setToasts(prev => prev.map(t => t.id === progressToastId ? { ...t, progress: pct } : t));
      });

      setUploading(false);
      // Mark progress toast complete then replace with success
  setToasts(prev => prev.map(t => t.id === progressToastId ? { ...t, progress: 100, inProgress: false, message: 'Queued for background classification...', variant: 'info-progress' } : t));
      // Brief delay to show 100% before success
      setTimeout(() => {
        setToasts(prev => prev.filter(t => t.id !== progressToastId));
        pushToast({ message: 'Bet uploaded (classification will appear shortly)', variant: 'success' });
      }, 600);
      setFile(null);
      setPreview(null);
  setBetData({ amount: 0 });

  // Refresh recent bets after successful upload
      await loadRecentBets();
      // Schedule a delayed refresh to pick up classification result from background worker
      setTimeout(() => { loadRecentBets(); }, 4000);
    } catch (err) {
      setUploading(false);
      let baseMsg = 'Error uploading bet.';
      let detail: string | null = null;
      if (err && typeof err === 'object') {
        const anyErr: any = err;
        if (anyErr.response) {
          const status = anyErr.response.status;
            const respData = typeof anyErr.response.data === 'string' ? anyErr.response.data : JSON.stringify(anyErr.response.data);
          detail = `Status ${status}${respData ? ' - ' + respData : ''}`;
        } else if (anyErr.message) {
          detail = anyErr.message;
        }
      }
      setError(baseMsg);
      setErrorDetails(detail);
  pushToast({
        message: detail ? `${baseMsg} ${detail}` : baseMsg,
        variant: 'danger',
        retry: () => {
          if (file && !uploading) {
            const fakeEvt = { preventDefault: () => {} } as unknown as React.FormEvent;
            handleSubmit(fakeEvt);
          }
        }
      });
      console.error('Error:', err);
    }
  };

  const clearForm = () => {
    setFile(null);
    setPreview(null);
    setError(null);
  setBetData({ amount: 0 });
  };

  return (
    <div className="container mt-4">
      <h2>Upload Bet Slip</h2>

      {error && (
        <Alert variant="danger" dismissible onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {error && errorDetails && (
        <div className="mb-2">
          <Button variant="link" size="sm" className="p-0" onClick={() => setShowErrorDetails(s => !s)}>
            {showErrorDetails ? 'Hide details' : 'Show details'}
          </Button>
          <Collapse in={showErrorDetails}>
            <div className="mt-1 small text-muted preserve-whitespace">{errorDetails}</div>
          </Collapse>
        </div>
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
                  title="Select bet slip image"
                  placeholder="Select bet slip image"
                />
              </div>

              {preview && (
                <div className="text-center">
                  <img src={preview} alt="Preview" className="img-fluid mb-3 upload-preview-img" />
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
                      value={betData.amount}
                      onChange={handleInputChange}
                      required
                    />
                  </Form.Group>

                  {/* Date is managed by backend (PlacedAt) and shown in recent list */}

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
          <Card>
            <Card.Header className="d-flex justify-content-between align-items-center">
              <span>Recent Bets</span>
              <Button variant="outline-secondary" size="sm" onClick={loadRecentBets} disabled={loadingRecent}>
                {loadingRecent ? 'Refreshing...' : 'Refresh'}
              </Button>
            </Card.Header>
            <Card.Body>
              {recentBets.length === 0 ? (
                <p className="text-muted">No recent bets found</p>
              ) : (
                <div className="list-group">
                  {recentBets.map(bet => (
                    <div key={bet.id} className="list-group-item">
                      <div className="d-flex w-100 justify-content-between">
                        <h5 className="mb-1">£{bet.amount.toFixed(2)}</h5>
                        <small>{new Date(bet.placedAt).toLocaleDateString()}</small>
                      </div>
                      <p className="mb-1">
                        <span className="badge bg-info">
                          {bet.writerClassification || 'Not classified'}
                        </span>
                      </p>
                      <small>Customer: {bet.customerName || 'Unknown'}</small>
                    </div>
                  ))}
                </div>
              )}
            </Card.Body>
          </Card>
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
                      <ProgressBar now={t.progress || 0} animated={t.inProgress} label={`${t.progress || 0}%`} min={0} max={100} />
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
