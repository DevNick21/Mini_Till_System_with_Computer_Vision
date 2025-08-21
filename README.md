# Mini Till System

End-to-end system for tracking customer activity with handwriting-based slip classification, simple threshold monitoring, and alerts.

## Architecture Overview

This repo contains a .NET 8 Web API backend, a React SPA frontend, and a Python FastAPI service for handwriting classification.

### Backend (.NET 8)

- ASP.NET Core Web API (net8.0)
- Entity Framework Core + SQLite (code-first with migrations applied on startup)
- Serilog logging (console + rolling file under `Logs/`)
- CORS policy for `http://localhost:3000` (React dev) and `http://localhost:5113`

### Frontend (React)

- React (CRA) with TypeScript types and React Bootstrap
- Dev server runs on port 3000 and proxies API calls to 5113

### ML Service (Python)

- FastAPI app (`cv_service/classification_api.py`) exposing `/health`, `/classify-anonymous`, `/model-info`
- Runs on port 8001 by default

## Key Features

- Customer CRUD
- Bet records with slip image upload (base64 or multipart file)
- On-demand handwriting classification via the Python service
- Simple threshold rules and alert generation
- Dashboard stats (totals for customers, bets, alerts)

## Technology Stack

- Backend: .NET 8, ASP.NET Core, EF Core (SQLite)
- Frontend: React + CRA, Bootstrap
- Logging: Serilog (console + file)
- ML Integration: Python FastAPI service

## Project Structure

```
main/
├── Program.cs                  # App startup & DI
├── appsettings.json            # Configuration
├── Controllers/                # API endpoints (Bets, Customers, Alerts, System)
├── Data/                       # EF Core DbContext & factory
├── DTOs/                       # Request/response models
├── Models/                     # EF Core entities
├── Services/                   # Data, classification, threshold services
├── Migrations/                 # EF Core migrations
├── frontend/                   # React SPA (CRA)
├── cv_service/                 # Python FastAPI classification service
└── Logs/                       # Serilog rolling logs
```

## Getting Started

1) Backend (.NET 8)

- Restore and run
  - dotnet restore
  - dotnet run
- Default URL: http://localhost:5113 (HTTPS also available at 7183)
- Migrations are applied automatically at startup

2) Frontend (React)

- In `frontend/`
  - npm install
  - npm start
- Opens http://localhost:3000 with proxy to http://localhost:5113

3) ML Classification Service (Python)

- In `cv_service/`
  - pip install -r requirements.txt
  - python classification_api.py
- Health: http://localhost:8001/health

## Configuration

Backend configuration (`appsettings.json`):

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Data Source=betfred.db"
  },
  "ClassificationApi": {
    "BaseUrl": "http://localhost:8001/",
    "TimeoutSeconds": 60
  },
  "FileUpload": {
    "MaxFileSizeBytes": 10485760,
    "AllowedContentTypes": ["image/jpeg", "image/png"]
  },
  "Serilog": {
    "WriteTo": [
      { "Name": "Console" },
      { "Name": "File", "Args": { "path": "Logs/log-.txt", "rollingInterval": "Day" } }
    ]
  }
}
```

Notes:
- The backend reads `ClassificationApi:BaseUrl`; ensure the Python service is running there.
- JWT settings may be present in the file but authentication/authorization is not enabled in this build.

## API Endpoints (current)

Customers (`/api/customers`):
- GET /api/customers — List all
- GET /api/customers/{id} — Get by ID
- POST /api/customers — Create
- PUT /api/customers/{id} — Update
- DELETE /api/customers/{id} — Delete

Bets (`/api/bet`):
- GET /api/bet — List all
- GET /api/bet/{id} — Get by ID
- POST /api/bet — Create using base64 image (CreateBetRecordDto)
- PUT /api/bet/{id} — Update amount/customer/classification fields
- DELETE /api/bet/{id} — Delete
- POST /api/bet/upload — Upload a slip file to create a bet (multipart/form-data; optional amount, customerId)
- POST /api/bet/{id}/upload-slip — Upload a slip file for an existing bet
- GET /api/bet/{id}/slip-image — Get the slip image bytes
- GET /api/bet/recent — 10 most recent bets
- GET /api/bet/{id}/status — Classification status for a bet

Alerts & Dashboard:
- GET /api/alerts — List alerts
- GET /api/alerts/dashboard — Dashboard stats

System utilities (`/api/system`):
- POST /api/system/create-demo-data — Seed demo bet records from `cv_service/slips`
- POST /api/system/reset-database — Drop/create database
- POST /api/system/create-default-rules — Seed basic threshold rules
- GET  /api/system/dashboard-stats — Dashboard stats
- GET  /api/system/alerts — List alerts

## Database Models

- Customers — Customer entity (Id, Name)
- BetRecords — Amount, PlacedAt, ImageData, optional CustomerId, classification fields
- Alerts — Threshold breach notifications
- ThresholdRules — Simple numeric rules (Name, Value, TimeWindowMinutes, IsActive)

## ML Service

- Endpoints: `/health`, `/classify-anonymous` (multipart file param `file`), `/model-info`
- Returns writerId (1-based) and confidence (0.0–1.0); backend stores values on the bet record

## Logs

- Located in `Logs/` with rolling daily files: `log-YYYYMMDD.txt`

## Contributing

1. Fork the repo
2. Create a feature branch
3. Keep changes focused and add tests when possible
4. Open a PR

## License

MIT License
