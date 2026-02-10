import os
import base64
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from .config import Config
from .api_client import get_prediction, check_model_health, ModelAPIError
from .utils import allowed_file, validate_file_size, generate_unique_filename
from .database import (
    get_supabase_client, create_hospital, get_hospital, get_all_hospitals,
    create_upload, create_analysis, create_patient_metadata,
    get_hospital_stats, get_global_stats, get_regional_data, create_alert,
    get_active_alerts
)


app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')
app.secret_key = Config.SECRET_KEY
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=Config.SESSION_LIFETIME_HOURS)
app.config['SESSION_COOKIE_SECURE'] = False  # Set True only when serving over HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'


# ===================== UTILITY ROUTES =====================

@app.route('/health')
def health():
    """Health check endpoint for Railway monitoring."""
    model_status = 'connected' if check_model_health() else 'disconnected'

    return jsonify({
        'status': 'healthy',
        'model_api': model_status,
        'timestamp': datetime.now().isoformat()
    })


# ===================== HOME / LANDING PAGE =====================

@app.route('/')
def index():
    """Landing page - choose between hospital portal or surveillance."""
    return render_template('index.html')


# ===================== HOSPITAL PORTAL ROUTES =====================

@app.route('/hospital/dashboard')
def hospital_dashboard():
    """Hospital portal dashboard."""
    if 'hospital_id' not in session:
        return redirect(url_for('hospital_login'))

    hospital_id = session['hospital_id']

    # Demo mode - provide mock hospital data
    hospital = {
        'id': hospital_id,
        'name': session.get('hospital_name', 'Demo Hospital'),
        'city': 'Demo City',
        'state': 'Demo State',
        'country': 'Demo Country',
    }

    # Mock stats for today
    stats_today = {
        'case_count': 127,
        'normal_count': 95,
        'pneumonia_count': 32,
        'severe_count': 5,
        'deaths': 0,
        'avg_confidence': 0.87,
    }

    # Mock stats for week
    stats_week = {
        'case_count': 845,
        'normal_count': 620,
        'pneumonia_count': 225,
        'severe_count': 38,
        'deaths': 2,
        'avg_confidence': 0.85,
    }

    return render_template('hospital/dashboard.html',
        hospital=hospital,
        stats_today=stats_today,
        stats_week=stats_week
    )


@app.route('/hospital/login', methods=['GET', 'POST'])
def hospital_login():
    """Hospital login page - Dummy authentication for demo purposes."""
    if request.method == 'POST':
        hospital_id = request.form.get('hospital_id') or 'demo-hospital'

        # Dummy authentication - accepts any input
        session.permanent = True  # Use configured session lifetime
        session['hospital_id'] = hospital_id
        session['hospital_name'] = 'Demo Hospital'
        return redirect(url_for('hospital_dashboard'))

    return render_template('hospital/login.html')


@app.route('/hospital/logout')
def hospital_logout():
    """Hospital logout."""
    session.clear()
    return redirect(url_for('index'))


@app.route('/hospital/upload', methods=['GET', 'POST'])
def hospital_upload():
    """Hospital upload interface."""
    if 'hospital_id' not in session:
        if request.method == 'POST':
            return jsonify({'error': 'Session expired. Please log in again.'}), 401
        return redirect(url_for('hospital_login'))

    if request.method == 'POST':
        if 'images' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('images')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400

        hospital_id = session['hospital_id']

        # Try to create upload record in database, fall back to demo mode
        upload_id = None
        use_demo_mode = False
        try:
            upload = create_upload(
                hospital_id=hospital_id,
                user_id=session.get('user_id', 'doctor-001'),
                image_count=len(files)
            )
            upload_id = upload['id'] if upload else None
        except Exception:
            use_demo_mode = True
            import uuid
            upload_id = str(uuid.uuid4())

        results = []
        for file in files:
            if not file or not file.filename:
                continue
            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'error': 'Invalid file type. Only JPG and PNG are allowed.'
                })
                continue
            if not validate_file_size(file):
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'error': 'File too large. Maximum size is 10MB.'
                })
                continue

            try:
                # Get prediction from model API
                prediction = get_prediction(file)

                pred_result = prediction.get('prediction', 'UNCERTAIN')
                conf_result = prediction.get('confidence', 0)
                severity = get_severity_from_confidence(conf_result)

                # Try to save to database if available
                if not use_demo_mode:
                    try:
                        analysis = create_analysis(
                            upload_id=upload_id,
                            image_path=f"uploads/{generate_unique_filename(file.filename)}",
                            prediction=pred_result,
                            confidence=conf_result,
                            severity=severity,
                            processing_time_ms=prediction.get('processing_time_ms', 0),
                            model_version=prediction.get('model_version', 'v1.0'),
                            heatmap_path=None
                        )

                        # Store patient metadata if provided
                        age_range = request.form.get('age_range', 'unknown')
                        gender = request.form.get('gender', 'Unknown')
                        vaccination = request.form.get('vaccination_status', 'unknown')
                        symptoms = request.form.getlist('symptoms')

                        create_patient_metadata(
                            analysis_id=analysis['id'],
                            age_range=age_range,
                            gender=gender,
                            vaccination_status=vaccination,
                            symptoms=symptoms
                        )
                    except Exception:
                        pass  # Database save failed, but prediction still works

                results.append({
                    'filename': file.filename,
                    'status': 'success',
                    'prediction': pred_result,
                    'confidence': conf_result,
                    'severity': severity,
                    'processing_time_ms': prediction.get('processing_time_ms', 0),
                    'model_version': prediction.get('model_version', 'v1.0'),
                    'heatmap': prediction.get('heatmap'),
                    'probabilities': prediction.get('probabilities', {})
                })

            except ModelAPIError as e:
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'error': str(e)
                })
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'error': 'An unexpected error occurred during analysis.'
                })

        return jsonify({
            'upload_id': upload_id,
            'results': results
        })

    return render_template('hospital/upload.html')


@app.route('/hospital/results/<upload_id>')
def hospital_results(upload_id):
    """View results for a specific upload."""
    if 'hospital_id' not in session:
        return redirect(url_for('hospital_login'))

    # Fetch results from database
    supabase = get_supabase_client()
    response = supabase.table('analyses') \
        .select('*') \
        .eq('upload_id', upload_id) \
        .execute()

    analyses = response.data

    return render_template('hospital/results.html', analyses=analyses)


# ===================== SURVEILLANCE DASHBOARD ROUTES =====================

@app.route('/surveillance')
def surveillance_dashboard():
    """Main surveillance dashboard - real-time outbreak tracking."""

    # Demo mode - provide mock data instead of database calls
    global_stats = {
        'case_count': 547000000,
        'normal_count': 420000000,
        'pneumonia_count': 127000000,
        'severe_count': 5470000,
        'deaths': 2735000,
    }

    # Mock regional data for map
    regional_data = [
        {'region_id': 'US', 'case_count': 450000, 'severe_count': 24000},
        {'region_id': 'IN', 'case_count': 800000, 'severe_count': 45000},
        {'region_id': 'BR', 'case_count': 300000, 'severe_count': 18000},
        {'region_id': 'GB', 'case_count': 142000, 'severe_count': 8500},
        {'region_id': 'FR', 'case_count': 128000, 'severe_count': 7200},
    ]

    # Mock active alerts
    alerts = [
        {
            'severity': 'critical',
            'region_id': 'Maharashtra, India',
            'description': 'Surge of 15,000 new cases in 24h. Hospital capacity at 92%.'
        },
        {
            'severity': 'high',
            'region_id': 'São Paulo, Brazil',
            'description': 'ICU occupancy reached 85%. Resource allocation recommended.'
        },
        {
            'severity': 'medium',
            'region_id': 'California, USA',
            'description': 'Unusual cluster detected in Bay Area. 450 cases in last 48h.'
        },
    ]

    return render_template('surveillance/dashboard.html',
        global_stats=global_stats,
        regional_data=regional_data,
        alerts=alerts,
        mapbox_token=Config.MAPBOX_ACCESS_TOKEN
    )


@app.route('/surveillance/region/<region_type>/<region_id>')
def surveillance_region(region_type, region_id):
    """Regional breakdown view."""
    supabase = get_supabase_client()

    # Get regional data
    response = supabase.table('regional_summary') \
        .select('*') \
        .eq('region_type', region_type) \
        .eq('region_id', region_id) \
        .order('date', desc=True) \
        .limit(30) \
        .execute()

    regional_data = response.data

    # Get sub-regions
    if region_type in ['country', 'state']:
        sub_type = 'state' if region_type == 'country' else 'city'
        sub_response = supabase.table('regional_summary') \
            .select('*') \
            .eq('region_type', sub_type) \
            .order('case_count', desc=True) \
            .limit(10) \
            .execute()
        sub_regions = sub_response.data
    else:
        sub_regions = []

    return render_template('surveillance/region.html',
        region_type=region_type,
        region_id=region_id,
        regional_data=regional_data,
        sub_regions=sub_regions
    )


@app.route('/surveillance/alerts')
def surveillance_alerts():
    """View all alerts."""
    # Demo mode - provide mock alerts
    alerts = [
        {
            'id': '1',
            'severity': 'critical',
            'region_id': 'Maharashtra, India',
            'alert_type': 'surge',
            'description': 'Surge of 15,000 new cases in 24h. Hospital capacity at 92%.',
            'triggered_at': '2026-02-10T08:30:00',
            'resolved_at': None,
        },
        {
            'id': '2',
            'severity': 'high',
            'region_id': 'São Paulo, Brazil',
            'alert_type': 'capacity',
            'description': 'ICU occupancy reached 85%. Resource allocation recommended.',
            'triggered_at': '2026-02-10T06:15:00',
            'resolved_at': None,
        },
        {
            'id': '3',
            'severity': 'medium',
            'region_id': 'California, USA',
            'alert_type': 'cluster',
            'description': 'Unusual cluster detected in Bay Area. 450 cases in last 48h.',
            'triggered_at': '2026-02-09T22:45:00',
            'resolved_at': None,
        },
        {
            'id': '4',
            'severity': 'high',
            'region_id': 'London, United Kingdom',
            'alert_type': 'surge',
            'description': 'Rapid increase in severe cases. 320% increase week-over-week.',
            'triggered_at': '2026-02-09T14:20:00',
            'resolved_at': None,
        },
        {
            'id': '5',
            'severity': 'critical',
            'region_id': 'Jakarta, Indonesia',
            'alert_type': 'capacity',
            'description': 'Hospital capacity exceeded. Emergency overflow protocols activated.',
            'triggered_at': '2026-02-09T11:00:00',
            'resolved_at': None,
        },
    ]

    return render_template('surveillance/alerts.html', alerts=alerts)


@app.route('/surveillance/hospitals')
def surveillance_hospitals():
    """View all participating hospitals and their status."""
    # Demo mode - provide mock hospital data
    hospital_data = [
        {
            'id': 'hosp-001',
            'name': 'City General Hospital',
            'city': 'Mumbai',
            'state': 'Maharashtra',
            'country': 'India',
            'current_cases': 245,
            'severe_cases': 18,
            'total_beds': 500,
            'icu_beds': 50,
        },
        {
            'id': 'hosp-002',
            'name': 'Metropolitan Medical Center',
            'city': 'São Paulo',
            'state': 'São Paulo',
            'country': 'Brazil',
            'current_cases': 312,
            'severe_cases': 24,
            'total_beds': 650,
            'icu_beds': 75,
        },
        {
            'id': 'hosp-003',
            'name': 'Bay Area Regional Hospital',
            'city': 'San Francisco',
            'state': 'California',
            'country': 'USA',
            'current_cases': 89,
            'severe_cases': 7,
            'total_beds': 400,
            'icu_beds': 45,
        },
        {
            'id': 'hosp-004',
            'name': 'Royal London Hospital',
            'city': 'London',
            'state': 'England',
            'country': 'United Kingdom',
            'current_cases': 156,
            'severe_cases': 12,
            'total_beds': 550,
            'icu_beds': 60,
        },
        {
            'id': 'hosp-005',
            'name': 'Central Jakarta Hospital',
            'city': 'Jakarta',
            'state': 'Jakarta',
            'country': 'Indonesia',
            'current_cases': 421,
            'severe_cases': 35,
            'total_beds': 450,
            'icu_beds': 40,
        },
    ]

    return render_template('surveillance/hospitals.html', hospitals=hospital_data)


# ===================== API ENDPOINTS =====================

@app.route('/api/v1/global-stats')
def api_global_stats():
    """API endpoint for global statistics."""
    days = request.args.get('days', default=1, type=int)
    # Demo mode - provide mock stats
    stats = {
        'case_count': 547000000,
        'normal_count': 420000000,
        'pneumonia_count': 127000000,
        'severe_count': 5470000,
        'deaths': 2735000,
    }
    return jsonify(stats)


@app.route('/api/v1/regional-data')
def api_regional_data():
    """API endpoint for regional data (for map)."""
    region_type = request.args.get('type', default='country')
    # Demo mode - provide mock regional data
    data = [
        {'region_id': 'US', 'case_count': 450000, 'severe_count': 24000},
        {'region_id': 'IN', 'case_count': 800000, 'severe_count': 45000},
        {'region_id': 'BR', 'case_count': 300000, 'severe_count': 18000},
        {'region_id': 'GB', 'case_count': 142000, 'severe_count': 8500},
        {'region_id': 'FR', 'case_count': 128000, 'severe_count': 7200},
    ]
    return jsonify(data)


@app.route('/api/v1/alerts')
def api_alerts():
    """API endpoint for alerts."""
    # Demo mode - provide mock alerts
    alerts = [
        {
            'severity': 'critical',
            'region_id': 'Maharashtra, India',
            'description': 'Surge of 15,000 new cases in 24h. Hospital capacity at 92%.'
        },
        {
            'severity': 'high',
            'region_id': 'São Paulo, Brazil',
            'description': 'ICU occupancy reached 85%. Resource allocation recommended.'
        },
        {
            'severity': 'medium',
            'region_id': 'California, USA',
            'description': 'Unusual cluster detected in Bay Area. 450 cases in last 48h.'
        },
    ]
    return jsonify(alerts)


@app.route('/api/v1/hospital/<hospital_id>/stats')
def api_hospital_stats(hospital_id):
    """API endpoint for hospital-specific statistics."""
    days = request.args.get('days', default=1, type=int)
    # Demo mode - provide mock stats
    stats = {
        'case_count': 127 * days,
        'normal_count': 95 * days,
        'pneumonia_count': 32 * days,
        'severe_count': 5 * days,
        'deaths': 0,
        'avg_confidence': 0.87,
    }
    return jsonify(stats)


# ===================== ERROR HANDLING =====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return render_template('error.html', error='Server error'), 500


# ===================== HELPER FUNCTIONS =====================

def get_severity_from_confidence(confidence: float) -> str:
    """Determine severity level based on confidence."""
    if confidence < 0.3:
        return 'mild'
    elif confidence < 0.7:
        return 'moderate'
    else:
        return 'severe'


if __name__ == '__main__':
    app.run(debug=Config.DEBUG, port=5000)
