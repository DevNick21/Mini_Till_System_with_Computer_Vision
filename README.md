# BetFred - Betting Slip Analysis System

A comprehensive betting slip analysis and monitoring system featuring handwritten slip classification, customer management, threshold monitoring, and automated alerting capabilities.

## Architecture Overview

This repository contains a full-stack application with three main components:

### Backend (.NET 8 Web API)

- **Framework**: ASP.NET Core Web API (net8.0)
- **Database**: Entity Framework Core with SQLite (code-first approach)
- **Logging**: Serilog with console and rolling file outputs (`Logs/` directory)
- **CORS**: Configured for React development server and production builds
- **Health Checks**: Built-in health monitoring endpoints
- **Background Services**: Automated threshold monitoring and alert generation

### Frontend (React SPA)

- **Framework**: React 19.1.0 with TypeScript
- **UI Library**: React Bootstrap 2.10.10 for responsive design
- **Routing**: React Router DOM for client-side navigation
- **State Management**: Custom hooks for async operations and event handling
- **Development**: Runs on port 3000 with proxy to backend API

### ML Classification Service (Python FastAPI)

- **Framework**: FastAPI for high-performance API endpoints
- **Model**: EfficientNet-B0 with custom classifier head for handwriting recognition
- **Image Processing**: OpenCV with CLAHE preprocessing pipeline
- **Training**: PyTorch with AdamW optimizer and early stopping
- **Deployment**: Uvicorn ASGI server on port 8001

## Key Features

### Core Functionality
- **Customer Management**: Full CRUD operations for customer records
- **Bet Record Management**: Upload and manage betting slips with image processing
- **Handwriting Classification**: AI-powered writer identification using EfficientNet-B0
- **Threshold Monitoring**: Configurable rules for automatic alert generation
- **Real-time Dashboard**: Live statistics and recent activity monitoring
- **Alert System**: Automated notifications for threshold breaches

### File Upload Support
- **Multiple Formats**: Support for JPEG and PNG image uploads
- **Upload Methods**: Both base64 encoding and multipart file uploads
- **File Size Limits**: Configurable maximum file size (10MB default)
- **Image Processing**: Automatic preprocessing with CLAHE enhancement

### Data Management
- **SQLite Database**: Lightweight, file-based database with automatic migrations
- **Entity Framework**: Code-first approach with seamless model updates
- **Background Processing**: Automated threshold evaluation and alert generation
- **Comprehensive Logging**: Structured logging with Serilog

## Technology Stack

### Backend Technologies
- **.NET 8**: Latest .NET framework with improved performance
- **ASP.NET Core**: High-performance web API framework
- **Entity Framework Core**: Modern ORM with SQLite database
- **AutoMapper**: Object-to-object mapping for DTOs
- **Serilog**: Structured logging with file and console outputs

### Frontend Technologies
- **React 19**: Latest React with concurrent features
- **TypeScript**: Type-safe JavaScript development
- **React Bootstrap**: Responsive UI components
- **React Router DOM**: Client-side routing
- **Axios**: HTTP client for API communication

### Machine Learning Stack
- **PyTorch 2.0.1**: Deep learning framework
- **torchvision**: Computer vision utilities and pre-trained models
- **OpenCV**: Image processing and computer vision
- **scikit-learn**: Machine learning utilities and metrics
- **FastAPI**: Modern, fast web framework for APIs

## Project Structure

```
bet_fred/
├── Program.cs                          # Application startup and dependency injection
├── appsettings.json                    # Configuration settings
├── bet_fred.csproj                    # .NET project file
├── betfred.db                         # SQLite database file
├── Controllers/                        # Web API controllers
│   ├── AlertsController.cs            # Alert management endpoints
│   ├── BetController.cs               # Bet record CRUD operations
│   ├── CustomersController.cs         # Customer management
│   ├── SystemController.cs            # System utilities and admin functions
│   └── ThresholdsController.cs        # Threshold configuration
├── Data/                              # Database context and configuration
│   ├── ApplicationDbContext.cs        # EF Core database context
│   └── DesignTimeDbContextFactory.cs  # Design-time factory for migrations
├── DTOs/                              # Data Transfer Objects
│   ├── BetRecordDto.cs                # Bet record request/response models
│   └── CustomerDto.cs                 # Customer request/response models
├── Models/                            # Entity models
│   ├── Alert.cs                       # Alert entity for notifications
│   ├── BetRecord.cs                   # Betting slip record entity
│   ├── Customer.cs                    # Customer entity
│   └── ThresholdRule.cs               # Threshold configuration entity
├── Services/                          # Business logic services
│   ├── ClassificationService.cs       # ML service integration
│   ├── DataService.cs                 # Data access layer
│   ├── ThresholdEvaluator.cs         # Threshold monitoring logic
│   └── ThresholdHostedService.cs     # Background service for monitoring
├── Migrations/                        # Entity Framework migrations
├── Mappings/                          # AutoMapper profiles
│   └── MappingProfile.cs              # DTO to entity mappings
├── Logs/                              # Application logs (rolling files)
├── frontend/                          # React SPA application
│   ├── src/
│   │   ├── components/                # React components
│   │   │   ├── alerts/               # Alert management UI
│   │   │   ├── customers/            # Customer management UI
│   │   │   ├── shared/               # Reusable components
│   │   │   └── upload/               # File upload components
│   │   ├── hooks/                     # Custom React hooks
│   │   ├── services/                  # API service layer
│   │   ├── types/                     # TypeScript type definitions
│   │   └── utils/                     # Utility functions
│   ├── package.json                   # Node.js dependencies
│   └── tsconfig.json                  # TypeScript configuration
├── cv_service/                        # Python ML classification service
│   ├── classification_api.py          # FastAPI application
│   ├── config.py                      # Configuration settings
│   ├── requirements.txt               # Python dependencies
│   ├── models/                        # ML model definitions
│   │   └── efficientnet_classifier.py # EfficientNet model implementation
│   ├── core/                          # Core ML functionality
│   │   └── inference.py               # Model inference utilities
│   ├── training/                      # Model training pipeline
│   │   ├── data_prep.py              # Data preparation and augmentation
│   │   ├── train_model.py            # Training loop implementation
│   │   └── evaluate_model.py         # Model evaluation utilities
│   ├── scripts/                       # Utility scripts
│   │   └── split_dataset.py          # Dataset splitting utility
│   ├── trained_models/                # Saved model weights and metadata
│   ├── dataset_splits/                # Training/validation/test splits
│   └── slips/                         # Training data images
└── slips_pdf/                         # Original PDF slip files
```

## Getting Started

### Prerequisites

- **.NET 8 SDK** or later
- **Node.js 16+** and npm
- **Python 3.11+** with pip
- **Git** for version control

### Quick Setup

#### 1. Backend (.NET 8 Web API)

```bash
# Navigate to project root
cd bet_fred

# Restore NuGet packages
dotnet restore

# Run the application
dotnet run
```

- **Development URL**: http://localhost:5113
- **HTTPS URL**: https://localhost:7183
- **Database**: SQLite migrations are applied automatically on startup
- **Logs**: Available in `Logs/` directory with daily rotation

#### 2. Frontend (React SPA)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

- **Development URL**: http://localhost:3000
- **Proxy Configuration**: Automatically proxies API calls to http://localhost:5113
- **Build Command**: `npm run build` for production builds

#### 3. ML Classification Service (Python FastAPI)

```bash
# Navigate to ML service directory
cd cv_service

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI service
python classification_api.py
```

- **Service URL**: http://localhost:8001
- **Health Check**: http://localhost:8001/health
- **API Documentation**: http://localhost:8001/docs (Swagger UI)

### Development Workflow

1. **Start all services** in separate terminal windows
2. **Access the application** at http://localhost:3000
3. **Upload betting slips** through the web interface
4. **Monitor alerts** and threshold breaches in real-time
5. **View logs** in the `Logs/` directory for debugging

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

## API Reference

### Customer Management (`/api/customers`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/customers` | List all customers |
| GET | `/api/customers/{id}` | Get customer by ID |
| POST | `/api/customers` | Create new customer |
| PUT | `/api/customers/{id}` | Update existing customer |
| DELETE | `/api/customers/{id}` | Delete customer |

### Bet Management (`/api/bet`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/bet` | List all bet records |
| GET | `/api/bet/{id}` | Get specific bet record |
| POST | `/api/bet` | Create bet with base64 image |
| PUT | `/api/bet/{id}` | Update bet record |
| DELETE | `/api/bet/{id}` | Delete bet record |
| POST | `/api/bet/upload` | Upload slip file to create bet |
| POST | `/api/bet/{id}/upload-slip` | Upload slip for existing bet |
| GET | `/api/bet/{id}/slip-image` | Retrieve slip image |
| GET | `/api/bet/recent` | Get 10 most recent bets |
| GET | `/api/bet/{id}/status` | Get classification status |

### Alert Management (`/api/alerts`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/alerts` | List all alerts |
| GET | `/api/alerts/dashboard` | Get dashboard statistics |

### Threshold Management (`/api/thresholds`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/thresholds` | List threshold rules |
| POST | `/api/thresholds` | Create threshold rule |
| PUT | `/api/thresholds/{id}` | Update threshold rule |
| DELETE | `/api/thresholds/{id}` | Delete threshold rule |

### System Utilities (`/api/system`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/system/create-demo-data` | Seed demo data from slip images |
| POST | `/api/system/reset-database` | Reset database (development only) |
| POST | `/api/system/create-default-rules` | Create default threshold rules |
| GET | `/api/system/dashboard-stats` | Get system statistics |
| GET | `/api/system/alerts` | Get system alerts |

### ML Service Endpoints (`cv_service`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health status |
| GET | `/model-info` | Model information and metrics |
| POST | `/classify-anonymous` | Classify uploaded slip image |

## Database Schema

### Core Entities

#### Customer
```csharp
public class Customer
{
    public int Id { get; set; }
    public string Name { get; set; }
    public DateTime CreatedAt { get; set; }
    public virtual ICollection<BetRecord> BetRecords { get; set; }
}
```

#### BetRecord
```csharp
public class BetRecord
{
    public int Id { get; set; }
    public decimal Amount { get; set; }
    public DateTime PlacedAt { get; set; }
    public byte[] ImageData { get; set; }
    public int? CustomerId { get; set; }
    public int? ClassificationWriterId { get; set; }
    public double? ClassificationConfidence { get; set; }
    public virtual Customer? Customer { get; set; }
}
```

#### Alert
```csharp
public class Alert
{
    public int Id { get; set; }
    public string Type { get; set; }
    public string Message { get; set; }
    public DateTime CreatedAt { get; set; }
    public bool IsResolved { get; set; }
}
```

#### ThresholdRule
```csharp
public class ThresholdRule
{
    public int Id { get; set; }
    public string Name { get; set; }
    public decimal Value { get; set; }
    public int TimeWindowMinutes { get; set; }
    public bool IsActive { get; set; }
}
```

## Machine Learning Pipeline

### Model Architecture
- **Base Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Custom Classifier**: Multi-layer perceptron with dropout and batch normalization
- **Input Processing**: CLAHE enhancement, grayscale conversion, normalization
- **Output**: Writer ID (1-based) and confidence score (0.0-1.0)

### Training Pipeline
- **Data Preparation**: Stratified train/validation splits
- **Augmentations**: Rotation, scaling, contrast adjustments
- **Optimization**: AdamW optimizer with learning rate scheduling
- **Early Stopping**: Prevents overfitting with configurable patience
- **Checkpointing**: Automatic saving of best model weights

### Inference API
- **Health Check**: `/health` - Service status and model information
- **Classification**: `/classify-anonymous` - Upload image for writer identification
- **Model Info**: `/model-info` - Model metadata and performance metrics

## Logging and Monitoring

### Application Logs
- **Location**: `Logs/` directory with daily rotation
- **Format**: Structured JSON logging with Serilog
- **Levels**: Information, Warning, Error, Debug
- **Retention**: Configurable log file retention policy

### Health Checks
- **Database Connectivity**: Entity Framework health checks
- **ML Service**: Classification service availability
- **System Resources**: Memory and disk usage monitoring

## Development Guidelines

### Code Structure
- **Controllers**: Handle HTTP requests and responses
- **Services**: Implement business logic and external integrations
- **Models**: Entity Framework entities for database mapping
- **DTOs**: Data transfer objects for API contracts

### Testing Strategy
- **Unit Tests**: Business logic and service layer testing
- **Integration Tests**: API endpoint and database testing
- **ML Model Tests**: Classification accuracy and performance testing

### Deployment Considerations
- **Database Migrations**: Automatic application on startup
- **Configuration Management**: Environment-specific settings
- **Logging**: Centralized logging for production monitoring
- **Health Checks**: Endpoint monitoring and alerting

## Contributing

### Development Setup
1. **Fork the repository** and clone your fork
2. **Create a feature branch** from `main`
3. **Install dependencies** for all three services
4. **Run tests** before submitting changes
5. **Follow code style** conventions and add documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
