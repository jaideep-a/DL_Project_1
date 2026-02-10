# Testing Guide - MediAlert Pandemic Surveillance System

This guide covers testing the MediAlert application locally before deployment.

## Prerequisites

- Python 3.10+
- pip (Python package manager)
- Git
- An active Supabase project with credentials in `.env`
- The model API running at: `https://dl-project-1-kqcz.onrender.com/api/predict`

## Step 1: Set Up Local Environment

### 1.1 Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.2 Install Dependencies
```bash
pip install -r requirements.txt
```

### 1.3 Verify Dependencies
```bash
python3 -c "import flask; import supabase; import requests; print('✓ All dependencies installed')"
```

## Step 2: Initialize Database

### 2.1 Run Database Setup Script
The `.env` file already contains your Supabase credentials. Run:
```bash
python3 setup_database.py
```

This will create all database tables and populate sample data:
- 5 hospitals with sample metadata
- 30-day outbreak simulation data
- 5 sample alerts at different severity levels
- Regional data for tracking

**Alternative**: If the script fails, manually run the SQL:
1. Go to https://gdtcnuzanixrmxgedqqp.supabase.co
2. Click **SQL Editor** → **New Query**
3. Copy all content from `database_schema.sql`
4. Paste and click **Run**

### 2.2 Verify Database Connection
```bash
python3 -c "from app.database import get_supabase_client; print('✓ Supabase connection OK')"
```

## Step 3: Test API Connectivity

### 3.1 Health Check Endpoint
```bash
# In a new terminal while app is running:
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_api": "connected",
  "timestamp": "2026-02-10T..."
}
```

### 3.2 Verify Model API
```bash
python3 -c "from app.api_client import check_model_health; print('✓ Model API OK' if check_model_health() else '✗ Model API error')"
```

## Step 4: Run Application Locally

### 4.1 Start Flask Development Server
```bash
python3 -m flask --app app.main:app run --debug
```

The app should be available at: **http://localhost:5000**

### 4.2 Expected Startup Output
```
 * Serving Flask app 'app.main:app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

## Step 5: Test Each Page

### 5.1 Landing Page (Home)
**URL**: http://localhost:5000/
- Should show MediAlert branding
- Problem statement with COVID statistics
- Links to Hospital Portal and Surveillance

### 5.2 Hospital Portal - Login
**URL**: http://localhost:5000/hospital/login
- **Username**: admin  (from Supabase hospitals table)
- **Password**: hospital123
- Should redirect to `/hospital/dashboard` on success
- Try with invalid credentials - should show error

### 5.3 Hospital Portal - Dashboard
**URL**: http://localhost:5000/hospital/dashboard
- Shows hospital stats (if logged in)
- Displays: Cases Today, Severe Cases, Normal Cases, Avg Confidence
- Shows 7-day case trends chart
- Recent analyses table (initially empty)
- Logout button in navigation

### 5.4 Hospital Portal - Upload
**URL**: http://localhost:5000/hospital/upload (requires login)
- Drag-and-drop upload interface
- Image preview
- Patient metadata collection:
  - Age range dropdown
  - Gender selection
  - Vaccination status
  - Symptoms (multi-select)
- Upload button (will call model API)
- Redirects to results page on success

### 5.5 Hospital Portal - Results
**URL**: http://localhost:5000/hospital/results/<upload_id>
- Shows AI analysis results
- Confidence score and severity badge
- Recommendation text
- Clinical guidance
- Summary statistics

### 5.6 Surveillance Dashboard - Main
**URL**: http://localhost:5000/surveillance
- **No login required** (public access)
- Top navigation bar with buttons:
  - "Hospital Portal" → links to `/hospital/login`
  - "Surveillance" → active page
- Displays sample pandemic data:
  - Global metrics: 2.5M cases, 45K new, 312K severe, 18.5K deaths
  - 30-day case trends chart (exponential growth pattern)
  - Geographic distribution section
  - Top affected regions (US, India, Brazil, UK)
  - Active alerts (3 critical/high/medium alerts)
  - Hospital network status (4,250 reporting)

### 5.7 Surveillance - Alerts Page
**URL**: http://localhost:5000/surveillance/alerts
- List of active alerts by severity
- Color-coded alert boxes
- Sample alerts for different cities

### 5.8 API Endpoints
Test these with curl or Postman:

**Global Statistics**:
```bash
curl http://localhost:5000/api/v1/global-stats
```

**Regional Data**:
```bash
curl http://localhost:5000/api/v1/regional-data
```

**Active Alerts**:
```bash
curl http://localhost:5000/api/v1/alerts
```

**Hospital Stats**:
```bash
curl http://localhost:5000/api/v1/hospital/hospital_001/stats
```

## Step 6: Common Issues & Troubleshooting

### Issue: "Module not found" when starting Flask
**Solution**:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Supabase connection error
**Solution**:
- Verify `.env` file has correct `SUPABASE_URL` and `SUPABASE_KEY`
- Check that database_schema.sql has been run in Supabase SQL Editor
- Verify internet connection

### Issue: Model API shows "disconnected"
**Solution**:
- Verify MODEL_API_URL is correct in `.env`: `https://dl-project-1-kqcz.onrender.com/api/predict`
- Check that model API server is running
- May take a few seconds to boot if on Render free tier

### Issue: Hospital login not working
**Solution**:
- Use test credentials: Username=`admin`, Password=`hospital123`
- Verify database was initialized (run setup_database.py)
- Check Supabase hospitals table contains sample data

### Issue: Upload returns 500 error
**Solution**:
- Ensure uploads/ directory exists: `mkdir -p uploads/`
- Verify model API is responding: `curl https://dl-project-1-kqcz.onrender.com/health`
- Check Flask debug output for detailed error message

## Step 7: Performance Testing

### Load Test Home Page
```bash
# Install Apache Bench if not present
ab -n 100 -c 10 http://localhost:5000/
```

### Load Test API Endpoint
```bash
ab -n 100 -c 10 http://localhost:5000/api/v1/global-stats
```

## Step 8: Code Quality Check

### Check Python Syntax
```bash
python3 -m py_compile app/*.py
```

### Run Basic Tests (if available)
```bash
pytest tests/ -v
```

## Testing Checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed successfully
- [ ] Database initialized with sample data
- [ ] Model API connectivity verified
- [ ] Flask server starts without errors
- [ ] Landing page loads with styling
- [ ] Hospital login works with test credentials
- [ ] Hospital dashboard displays (after login)
- [ ] Upload interface loads with drag-drop
- [ ] Surveillance dashboard displays without login
- [ ] Top navigation buttons work
- [ ] API endpoints respond with JSON
- [ ] No browser console errors
- [ ] Charts render correctly
- [ ] All links work (no 404 errors)

## Next Steps

Once testing is complete:

1. **Local Testing Complete**: All pages work locally ✓
2. **Deploy to Railway**:
   - Push to GitHub: `git push origin claude/review-repo-changes-404Rt`
   - Create Pull Request on GitHub
   - Connect Railway to GitHub repository
   - Add environment variables in Railway dashboard
   - Deploy

See `READY_TO_DEPLOY.md` for deployment instructions.

## Support

For issues during testing:
1. Check Flask debug output for error messages
2. Verify `.env` credentials are correct
3. Check database connection with: `python3 -c "from app.database import get_supabase_client; get_supabase_client()"`
4. Review logs in Railway dashboard after deployment
