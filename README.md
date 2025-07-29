# BetFred Smart Customer Behaviour Tracker

A comprehensive customer behavior tracking system for fraud detection and handwriting analysis.

## Architecture Overview

This application provides a complete fraud detection system with handwriting analysis capabilities, threshold monitoring, and real-time alerts.

### Backend (.NET 8)

- **Entity Framework Core** - Database operations and migrations
- **ASP.NET Core Web API** - RESTful API endpoints
- **Background Services** - Automated threshold monitoring
- **Serilog Logging** - Structured logging with file output

### Frontend (React)

- **React with TypeScript** - Modern component-based UI
- **React Router** - Client-side navigation
- **React Bootstrap** - Responsive UI components
- **Axios** - API client for backend communication
- **File Upload** - Drag & drop image processing

## Key Features

- 👥 **Customer Management** - Complete CRUD operations for customer records
- 📊 **Dashboard** - Real-time statistics and system monitoring
- 🔍 **Classification System** - Machine learning-powered handwriting analysis
- 🚨 **Alert System** - Configurable threshold-based monitoring
- 📁 **File Upload** - PDF and image processing for betting slips
- 🔐 **JWT Authentication** - Secure API access with token management
- 📈 **Analytics** - Clustering and pattern analysis
- ⚙️ **Configuration** - Flexible threshold rules and system settings

## Technology Stack

- **Backend**: .NET 9, ASP.NET Core, Entity Framework Core
- **Database**: SQLite with Entity Framework migrations
- **Frontend**: JavaScript, HTML5, CSS3, Bootstrap
- **Authentication**: JWT Bearer tokens
- **Logging**: Serilog with console and file sinks
- **ML Integration**: Python-based classification service
- **File Processing**: PDF and image analysis capabilities
- **Authentication**: JWT Bearer tokens
- **Logging**: Serilog with file output

## Project Structure

```
bet_fred/
├── Program.cs                    # Application entry point and configuration
├── appsettings.json             # Application configuration
├── bet_fred.csproj              # Project dependencies and settings
├── Controllers/
│   ├── BetController.cs         # Bet record management endpoints
│   ├── ClassificationController.cs # ML classification endpoints
│   └── CustomersController.cs   # Customer CRUD operations
├── Data/
│   └── ApplicationDbContext.cs  # Entity Framework database context
├── Models/
│   ├── Alert.cs                 # Alert entity and data model
│   ├── BetRecord.cs            # Bet record entity
│   ├── Customer.cs             # Customer entity
│   ├── PendingTag.cs           # Pending tag entity
│   ├── ThresholdRule.cs        # Threshold rule configuration
│   └── WriterClassification.cs # Classification result entity
├── Services/
│   ├── ClassificationService.cs # ML integration service
│   ├── DataService.cs          # Core business logic
│   ├── ThresholdEvaluator.cs   # Alert threshold evaluation
│   └── ThresholdHostedService.cs # Background monitoring service
├── Middleware/
│   └── GlobalExceptionMiddleware.cs # Global error handling
├── Migrations/                  # Entity Framework database migrations
├── cv_service/                 # Python ML classification service
│   ├── classification_api.py   # ML API endpoint
│   └── requirements.txt        # Python dependencies
├── wwwroot/                    # Static web assets
│   ├── assets/
│   │   ├── css/
│   │   │   └── dashboard.css   # Application styling
│   │   └── js/
│   │       ├── api.js          # API communication utilities
│   │       ├── customers.js    # Customer management functionality
│   │       └── dashboard.js    # Dashboard functionality
│   ├── clusters.html           # Cluster analysis page
│   ├── customers.html          # Customer management page
│   ├── dashboard.html          # Main dashboard
│   ├── index.html             # Application entry point
│   ├── navbar.html            # Navigation component
│   ├── rules.html             # Threshold rules configuration
│   ├── tag.html               # Tagging interface
│   └── upload.html            # File upload interface
└── Logs/                       # Application log files
```

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/DevNick21/Betfred-Smart-Customer-Behaviour-Tracker.git
   cd bet_fred
   ```

2. **Install dependencies**

   ```bash
   dotnet restore
   ```

3. **Set up the Python ML service**

   ```bash
   cd cv_service
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   dotnet run
   ```

5. **Access the application**
   - Main Dashboard: `http://localhost:5000`
   - Customer Management: `http://localhost:5000/customers.html`
   - File Upload: `http://localhost:5000/upload.html`
   - Rules Configuration: `http://localhost:5000/rules.html`

## API Endpoints

### Customers

- `GET /api/customers` - Get all customers
- `GET /api/customers/{id}` - Get customer by ID
- `POST /api/customers` - Create new customer
- `PUT /api/customers/{id}` - Update customer
- `DELETE /api/customers/{id}` - Delete customer

### Bet Records

- `GET /api/bet` - Get all bet records
- `POST /api/bet` - Create new bet record
- `GET /api/bet/{id}` - Get bet record by ID

### Classifications

- `POST /api/classification/analyze` - Analyze handwriting
- `GET /api/classification/cluster` - Get cluster data
- `POST /api/classification/tag` - Tag classification results

### Alerts & Rules

- `GET /api/alerts` - Get system alerts
- `GET /api/rules` - Get threshold rules
- `POST /api/rules` - Create new threshold rule

## Configuration

Update `appsettings.json` for your environment:

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Data Source=betfred.db"
  },
  "JwtSettings": {
    "Key": "your-secret-key-here",
    "Issuer": "BetFred",
    "Audience": "BetFredApp",
    "ExpirationHours": 24
  },
  "ClassificationApiSettings": {
    "PythonScriptPath": "cv_service/classification_api.py",
    "MaxFileSize": 10485760,
    "AllowedExtensions": [".pdf", ".jpg", ".jpeg", ".png"]
  },
  "ThresholdSettings": {
    "EvaluationIntervalMinutes": 5,
    "MaxAlerts": 1000
  }
}
```

## Database

### Key Tables

- **Customers** - Customer information and metadata
- **BetRecords** - Betting slip records with image data
- **Alerts** - System alerts and notifications
- **ThresholdRules** - Configurable monitoring rules
- **WriterClassifications** - ML classification results
- **PendingTags** - Items awaiting manual classification
- **WriterClassifications** - ML classification results
- **PendingTags** - Items awaiting manual classification

## Machine Learning Integration

The system includes a Python-based classification service for handwriting analysis:

- **Location**: `cv_service/`
- **API**: `classification_api.py`
- **Models**: Pre-trained models in `trained_models/`
- **Processing**: Supports PDF and image file analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code patterns
4. Test your changes thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
