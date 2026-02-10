# MediAlert - Pneumonia Detection & Pandemic Surveillance Platform

A deep learning-powered web application for analyzing chest X-ray images to detect pneumonia and track pandemic outbreaks in real time. Built to support hospitals and public health authorities in monitoring, detecting, and responding to respiratory disease outbreaks such as pneumonia pandemics.

## Project Overview

MediAlert addresses the critical need for rapid, scalable pneumonia detection during pandemic scenarios. By combining AI-driven chest X-ray analysis with a real-time surveillance dashboard, the platform enables:

- **Early outbreak detection** through automated X-ray screening at scale
- **Real-time pandemic monitoring** with global and regional dashboards
- **Hospital resource coordination** by tracking case loads, ICU capacity, and severity trends
- **Alert-based response** with automated notifications for case surges and capacity thresholds

## Features

- Upload chest X-ray images via drag-and-drop or file picker
- Get instant pneumonia predictions with confidence scores from a deep learning model
- View analysis heatmaps highlighting regions of concern
- Real-time pandemic surveillance dashboard with global/regional statistics
- Hospital portal for managing uploads and viewing patient analysis results
- Automated alert system for outbreak surges and hospital capacity warnings
- Responsive design for mobile and desktop
- RESTful API endpoints for data integration

## Technology Stack

- **Backend**: Python 3.10, Flask
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)
- **Database**: Supabase (PostgreSQL)
- **ML Model**: External deep learning model API for pneumonia detection (trained separately)
- **HTTP Client**: Requests
- **Deployment**: Railway (with Gunicorn)

## Architecture

```
User Browser --> Web App (Flask/Railway) --> Model API (External) --> Response --> User
                        |
                        v
                  Supabase (PostgreSQL)
                  (Patient data, alerts, surveillance)
```

The web application serves as the frontend and API layer. It forwards X-ray images to an external deep learning model API for inference and stores results in the database for surveillance and reporting.

## Project Structure

```
DL_Project_1/
|-- README.md                <- Description of project and how to set up and run it
|-- requirements.txt         <- Requirements file to document dependencies
|-- setup_database.py        <- Script to set up project database
|-- app/                     <- Main application package
|   |-- main.py              <- Flask routes and application entry point
|   |-- config.py            <- Configuration from environment variables
|   |-- api_client.py        <- Model API client for predictions
|   |-- database.py          <- Supabase database integration
|   |-- utils.py             <- Utility functions
|   |-- __init__.py
|-- scripts/                 <- Directory for pipeline scripts or utility scripts
|-- models/                  <- Directory for trained models
|-- data/                    <- Directory for project data
|   |-- raw/                 <- Raw data or script to download
|   |-- processed/           <- Processed data
|   |-- outputs/             <- Output data
|-- notebooks/               <- Directory for exploration notebooks (not graded)
|-- templates/               <- HTML templates (Jinja2)
|   |-- base.html
|   |-- index.html
|   |-- error.html
|   |-- hospital/            <- Hospital portal templates
|   |-- surveillance/        <- Surveillance dashboard templates
|-- static/                  <- Static assets
|   |-- css/
|   |-- js/
|-- tests/                   <- Unit tests
|-- docs/                    <- Documentation
|-- uploads/                 <- Upload directory for X-ray images
|-- Procfile                 <- Deployment config (Gunicorn)
|-- runtime.txt              <- Python version specification
|-- database_schema.sql      <- Database schema definition
|-- .env.example             <- Environment variable template
|-- .gitignore               <- Git ignore file
```

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DL_Project_1
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file (see `.env.example` for template):
   ```
   MODEL_API_URL=<your-model-api-url>/predict
   MODEL_API_HEALTH_URL=<your-model-api-url>/health
   MODEL_API_KEY=<optional-api-key>
   SECRET_KEY=<generate-a-random-key>
   SUPABASE_URL=<your-supabase-url>
   SUPABASE_KEY=<your-supabase-key>
   ```

5. Initialize the database:
   ```bash
   python setup_database.py
   ```

6. Run the development server:
   ```bash
   python -m flask --app app.main run --debug
   ```

7. Open http://localhost:5000

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_API_URL` | Yes | - | URL of the model prediction endpoint |
| `MODEL_API_HEALTH_URL` | No | - | URL for model API health checks |
| `MODEL_API_KEY` | No | - | API key if authentication is required |
| `API_TIMEOUT_SECONDS` | No | 10 | Timeout for API requests |
| `MAX_FILE_SIZE_MB` | No | 10 | Maximum upload file size |
| `SECRET_KEY` | No | dev-key | Flask session secret key |
| `DEBUG` | No | False | Enable debug mode |
| `SUPABASE_URL` | Yes | - | Supabase project URL |
| `SUPABASE_KEY` | Yes | - | Supabase API key |
| `MAPBOX_ACCESS_TOKEN` | No | - | Mapbox token for map visualizations |

## Model API Integration

The application expects the model API to follow this contract:

**Request:**
```
POST /predict
Content-Type: multipart/form-data
Body: image (file)
```

**Response:**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.87,
  "probabilities": {"NORMAL": 0.13, "PNEUMONIA": 0.87},
  "processing_time_ms": 245,
  "model_version": "v1.2",
  "heatmap": "<base64-encoded-image>"
}
```

See [docs/API_INTEGRATION.md](docs/API_INTEGRATION.md) for details.

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Railway deployment instructions.

## AI Usage Disclosure

This project used **Claude (Anthropic)** as an AI coding assistant for the **web development** components of the application. Specifically, Claude was used to help build:

- The Flask web application (`app/` directory: routes, configuration, API client, database layer, utilities)
- HTML/CSS/JS frontend templates and static assets (`templates/`, `static/`)
- Unit tests (`tests/`)
- Database schema and setup scripts
- Deployment configuration and documentation

The following merged pull requests were developed with Claude AI assistance:
- PR #5: Deployment guide and quick-start guide
- PR #6: Fix conflicting dependency versions
- PR #7: Database setup script
- PR #8: Testing and project summary documentation
- PR #9: Website interactivity and design improvements
- PR #10: X-ray upload UI and error handling
- PR #11: API connection and session configuration fixes

**Claude was NOT used for the deep learning model training.** The pneumonia detection model was developed and trained independently by the team.

## Team Members

- **Jaydip Patel** - Project lead, model training, web development
- **Hanfu** - Project member
- **Keming** - Project member

## Medical Disclaimer

This application is for **educational and research purposes only**. It is not intended for clinical diagnosis. Always consult qualified healthcare professionals for medical advice.

## License

MIT License
