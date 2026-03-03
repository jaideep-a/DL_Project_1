# MediAlert — Pneumonia Detection & Pandemic Surveillance Platform

A deep learning-powered web application for analyzing chest X-ray images to detect pneumonia and track pandemic outbreaks in real time. Built to support hospitals and public health authorities in monitoring, detecting, and responding to respiratory disease outbreaks.

---

## Project Overview

MediAlert combines AI-driven chest X-ray analysis with a real-time pandemic surveillance dashboard to enable:

- **Early outbreak detection** through automated X-ray screening at scale
- **Real-time pandemic monitoring** with global and regional dashboards
- **Hospital resource coordination** tracking case loads, ICU capacity, and severity trends
- **Automated alerts** for case surges and hospital capacity warnings

---

## Features

- Upload chest X-ray images via drag-and-drop or file picker
- Instant pneumonia predictions with confidence scores from an external deep learning model API
- Real-time pandemic surveillance dashboard with global/regional statistics
- Hospital portal for managing uploads and viewing analysis results
- Automated alert engine for outbreak surges and capacity warnings
- Predictive analytics: 7-day case forecasting and growth metrics
- RESTful API endpoints for data integration
- Responsive design for mobile and desktop

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, Flask 3.0 |
| Frontend | HTML5, CSS3, JavaScript (vanilla) |
| Database | Supabase (PostgreSQL) |
| ML Model | External deep learning API (pneumonia detection) |
| ML Analytics | scikit-learn, pandas, numpy |
| Deployment | Render (Gunicorn) |

---

## Architecture

```
User Browser
     │
     ▼
Web App (Flask / Render)
     │                  └──► Model API (external) ──► Prediction result
     │
     ▼
Supabase (PostgreSQL)
(hospital records, case summaries, alerts, surveillance data)
```

The web app forwards X-ray images to an external pneumonia detection model API and stores results in Supabase for surveillance and reporting.

---

## Project Structure

```
├── README.md               <- description of project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── Makefile                <- setup and run project from command line
├── setup.py                <- script to set up project (directories, deps, config check)
├── main.py                 <- main entry point to run the web application
│
├── website/                <- Flask web application
│   ├── __init__.py
│   ├── main.py             <- Flask routes and request handlers
│   ├── config.py           <- configuration loaded from environment variables
│   ├── api_client.py       <- HTTP client for external ML model API
│   ├── database.py         <- Supabase database integration layer
│   └── utils.py            <- file validation and utility helpers
│
├── models/                 <- ML model code (forecasting and alert engine)
│   ├── __init__.py
│   ├── predictions.py      <- CaseForecastModel, GrowthAnalyzer, ResourceDemandPredictor
│   └── alerts.py           <- AlertEngine (surge, capacity, growth alerts)
│
├── scripts/                <- pipeline and utility scripts
│   ├── make_dataset.py     <- verify / prepare chest X-ray dataset
│   ├── build_features.py   <- feature extraction pipeline
│   ├── model.py            <- train model / run offline predictions (CLI)
│   └── setup_database.py   <- initialize Supabase database schema
│
├── data/                   <- project data
│   ├── raw/                <- raw data (place chest_xray/ folder here)
│   ├── processed/          <- processed feature data
│   ├── outputs/            <- output data
│   └── database_schema.sql <- Supabase SQL schema definition
│
├── notebooks/              <- exploration notebooks (not graded)
│
├── templates/              <- Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── error.html
│   ├── results.html
│   ├── hospital/           <- hospital portal templates
│   │   ├── login.html
│   │   ├── dashboard.html
│   │   ├── upload.html
│   │   └── results.html
│   └── surveillance/       <- surveillance dashboard templates
│       ├── dashboard.html
│       ├── alerts.html
│       └── predictions.html
│
├── static/                 <- static assets
│   ├── css/style.css
│   └── js/app.js
│
├── tests/                  <- unit tests
│   ├── test_api_client.py
│   ├── test_utils.py
│   └── test_predictions.py
│
├── Procfile                <- Render/Gunicorn deployment config
├── runtime.txt             <- Python version (3.11.0)
├── .env.example            <- environment variable template
└── .gitignore
```

---

## Setup

### Prerequisites

- Python 3.11+
- pip
- A [Supabase](https://supabase.com) project (free tier works)
- External model API URL = https://dl-project-1-kqcz.onrender.com

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/jaydip12357/DL_Project_1.git
cd DL_Project_1
git checkout final_version
```

**2. Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
# or:
make install
```

**4. Configure environment variables:**
```bash
cp .env.example .env
# Edit .env and fill in your credentials
```

Required `.env` values:
```
MODEL_API_URL=https://dl-project-1-kqcz.onrender.com/api/predict
MODEL_API_HEALTH_URL=https://dl-project-1-kqcz.onrender.com
```

**5. Initialize the database:**
```bash
python scripts/setup_database.py
# or:
make setup
```

**6. Run the application:**
```bash
python main.py
# or:
make run
```

Open **http://localhost:5000** in your browser.

---

## Makefile Commands

```bash
make install   # pip install -r requirements.txt
make setup     # run setup.py + setup_database.py
make run       # python main.py
make test      # pytest tests/ -v
make clean     # remove __pycache__ and .pyc files
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_API_URL` | Yes | — | Prediction endpoint of the external ML model |
| `MODEL_API_HEALTH_URL` | No | — | Health check URL for the model API |
| `MODEL_API_KEY` | No | — | API key if the model requires authentication |
| `SUPABASE_URL` | Yes | — | Supabase project URL |
| `SUPABASE_KEY` | Yes | — | Supabase anon/service key |
| `SECRET_KEY` | No | dev-key | Flask session secret key |
| `DEBUG` | No | False | Enable Flask debug mode |
| `API_TIMEOUT_SECONDS` | No | 30 | Timeout for model API requests |
| `MAX_FILE_SIZE_MB` | No | 10 | Maximum upload file size |
| `MAPBOX_ACCESS_TOKEN` | No | — | Mapbox token for map visualizations |

---

## Model API Contract

The app sends X-ray images to an external model API. The API must accept:

**Request:**
```
POST /predict
Content-Type: multipart/form-data
Body: image (file field)
```

**Response:**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.87,
  "probabilities": {"NORMAL": 0.13, "PNEUMONIA": 0.87},
  "processing_time_ms": 245,
  "model_version": "v1.2",
  "heatmap": "<optional-base64-image>"
}
```

If the model API is unavailable, the app falls back to a deterministic demo result so the UI always responds.

---

## Dataset

The chest X-ray dataset used for model training is the [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset.

Place it under `data/raw/chest_xray/` with the following structure:
```
data/raw/chest_xray/
    train/   NORMAL/   PNEUMONIA/
    val/     NORMAL/   PNEUMONIA/
    test/    NORMAL/   PNEUMONIA/
```

Run `python scripts/make_dataset.py` to verify the layout.

---

## Testing

```bash
python -m pytest tests/ -v
```

Tests cover:
- `test_api_client.py` — model API request/response handling
- `test_utils.py` — file validation utilities
- `test_predictions.py` — forecasting models and alert engine

---

## Deployment (Render)

1. Push the `final_version` branch to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Connect to `https://github.com/jaydip12357/DL_Project_1`
4. Set **Branch** to `final_version`
5. Set **Build Command**: `pip install -r requirements.txt`
6. Set **Start Command**: `gunicorn website.main:app`
7. Add all required environment variables in the Render dashboard

---

## AI Usage Disclosure

This project used **Claude (Anthropic)** as an AI coding assistant for the web application components. Claude was used to help build:

- Flask web application (`website/` directory: routes, config, API client, database layer, utilities)
- HTML/CSS/JS frontend (`templates/`, `static/`)
- ML analytics module (`models/predictions.py`, `models/alerts.py`)
- Unit tests (`tests/`)
- Pipeline scripts (`scripts/`)
- Database schema and setup scripts
- Project structure and deployment configuration

**Claude was NOT used for training the deep learning pneumonia detection model.** The model was developed and trained independently by the team. Claude code was used for website development.

---

## Team Members

- **Jaideep Aher**
- **Hanfu**
- **Keming**

---

## Medical Disclaimer

This application is for **educational and research purposes only**. It is not intended for clinical diagnosis. Always consult qualified healthcare professionals for medical advice.

---

## License

MIT License
