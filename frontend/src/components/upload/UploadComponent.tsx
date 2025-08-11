import React, { useState, useEffect } from 'react';
import { Form, Button, Card, Alert, Spinner } from 'react-bootstrap';
import { apiService } from '../../services/api';
import { BetRecord } from '../../types';

const UploadComponent: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [betData, setBetData] = useState<Partial<BetRecord>>({
    amount: 0,
    odds: '',
    date: new Date().toISOString().split('T')[0],
    handwritingClassification: ''
  });
  const [recentBets, setRecentBets] = useState<BetRecord[]>([]);

  useEffect(() => {
    loadRecentBets();
  }, []);

  const loadRecentBets = async () => {
    try {
      const bets = await apiService.getRecentBets();
      setRecentBets(bets);
    } catch (error) {
      console.error('Error loading recent bets:', error);
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
      setUploadSuccess(false);
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
      setUploadSuccess(false);
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

  // Upload image along with stake (amount); backend will create the bet and return ID
  const betId = await apiService.uploadBetImage(file, betData.amount);
      
      // Now update the bet with the form data
  await apiService.updateBet(parseInt(betId), { id: parseInt(betId) });

      setUploading(false);
      setUploadSuccess(true);
      setFile(null);
      setPreview(null);
      setBetData({
        amount: 0,
        odds: '',
        date: new Date().toISOString().split('T')[0],
        handwritingClassification: ''
      });

      // Refresh recent bets
      loadRecentBets();
    } catch (err) {
      setUploading(false);
      setError('Error uploading bet. Please try again.');
      console.error('Error:', err);
    }
  };

  const clearForm = () => {
    setFile(null);
    setPreview(null);
    setError(null);
    setUploadSuccess(false);
    setBetData({
      amount: 0,
      odds: '',
      date: new Date().toISOString().split('T')[0],
      handwritingClassification: ''
    });
  };

  return (
    <div className="container mt-4">
      <h2>Upload Bet Slip</h2>

      {error && (
        <Alert variant="danger" dismissible onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {uploadSuccess && (
        <Alert variant="success" dismissible onClose={() => setUploadSuccess(false)}>
          Bet uploaded successfully!
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

                  <Form.Group className="mb-3">
                    <Form.Label>Odds</Form.Label>
                    <Form.Control
                      type="text"
                      name="odds"
                      value={betData.odds}
                      onChange={handleInputChange}
                      required
                    />
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>Date</Form.Label>
                    <Form.Control
                      type="date"
                      name="date"
                      value={betData.date}
                      onChange={handleInputChange}
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
          <Card>
            <Card.Header>Recent Bets</Card.Header>
            <Card.Body>
              {recentBets.length === 0 ? (
                <p className="text-muted">No recent bets found</p>
              ) : (
                <div className="list-group">
                  {recentBets.map(bet => (
                    <div key={bet.id} className="list-group-item">
                      <div className="d-flex w-100 justify-content-between">
                        <h5 className="mb-1">£{bet.amount.toFixed(2)}</h5>
                        <small>{new Date(bet.date).toLocaleDateString()}</small>
                      </div>
                      <p className="mb-1">Odds: {bet.odds}</p>
                      <p className="mb-1">
                        <span className="badge bg-info">
                          {bet.handwritingClassification || 'Not classified'}
                        </span>
                      </p>
                      <small>Customer: {bet.customer?.name || 'Unknown'}</small>
                    </div>
                  ))}
                </div>
              )}
            </Card.Body>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default UploadComponent;
