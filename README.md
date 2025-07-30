
# Betfred Smart Customer Behaviour Tracker

## Overview
This project demonstrates a proof-of-concept system for Betfred to:
- Ingest customer bet records and scanned slip images via a two-step API.
- Store metadata and raw image data in a SQLite database using EF Core.
- Classify handwriting styles across slips with a Python microservice.
- Associate clustered slips back to customers and simulate threshold-based alerts.


## Features
- **ASP.NET Core (.NET 8)** Web API with minimal-API and controller support
- **Entity Framework Core (8.0.13)** + **SQLite** for lightweight persistence
- **BLOB storage** of greyscale slip images in the database
- **Two-step ingestion:**
  1. Create bet metadata record (`POST /bets`)
  2. Upload slip image (`POST /bets/{id}/slip`)
- **Handwriting classifier** via an external Python microservice (FastAPI + scikit-learn)
- **Threshold rules** for automated alert generation
- **Swagger/OpenAPI** documentation and UI in development mode


## Prerequisites
- [.NET 8 SDK](https://dotnet.microsoft.com/download)
- [SQLite 3](https://www.sqlite.org/download.html) (optional – file is created automatically)
- [Python 3.10+](https://www.python.org/downloads/) with `fastapi`, `uvicorn`, `opencv-python`, `scikit-learn`
## Getting Started
### 1. Clone the repository
```bash
git clone https://your.git.repo/Betfred-Smart-Customer-Behaviour-Tracker.git
cd Betfred-Smart-Customer-Behaviour-Tracker
```

### 2. Configure
```JSON
{
  "ConnectionStrings": {
    "DefaultConnection": "Data Source=betfred.db"
  }
}
```

### 3. Apply database migrations
```bash
dotnet restore
dotnet ef migrations add InitialCreate
dotnet ef database update
```

### 4. Run the Web API
```bash
dotnet run
```
- Swagger UI available at https://localhost:5001/swagger

### 5. Prepare and run the Python classification service
```bash
cd cv_service
pip install fastapi uvicorn opencv-python scikit-learn
uvicorn main: app --reload
```
- Exposes `POST /cluster` for handwriting clustering
## Usage
### 1. Create a bet record (metadata only):
```http
POST /bets
Content-Type: application/json

{
  "amount": 10.0,
  "betType": "Win",
  "sport": "Tennis",
  "description": "Federer vs Nadal",
  "odds": 1.5,
  "customerId": 1
}
```
### 2. Upload the slip image:
```http
POST /bets/{id}/slip
Content-Type: multipart/form-data
Form-Field "slip": (binary JPEG/PNG)
```
### 3. Trigger clustering & alert simulation:
```csharp
// After uploading all slips:
var all = await db.BetRecords.Where(b => b.ImageData.Length > 0)
                              .Select(b => new { b.Id, b.ImageData })
                              .ToListAsync();
var clusters = await http.PostAsJsonAsync("http://localhost:8000/cluster", all);
// Save clusters and generate alerts based on ThresholdRules
```
## Project Structure
```bash
/Controllers      # API controllers (if any)
/Data             # ApplicationDbContext and EF Core setup
/Migrations       # EF Core migration files
/Models           # C# entity classes
/Program.cs       # App startup and endpoint wiring
/appsettings.json # Connection strings and config
/cv_service       # Python FastAPI clustering service
```
