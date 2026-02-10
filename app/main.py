import os
from datetime import datetime
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
    hospital = get_hospital(hospital_id)

    if not hospital:
        session.clear()
        return redirect(url_for('hospital_login'))

    # Get stats
    stats_today = get_hospital_stats(hospital_id, days=1)
    stats_week = get_hospital_stats(hospital_id, days=7)

    return render_template('hospital/dashboard.html',
        hospital=hospital,
        stats_today=stats_today,
        stats_week=stats_week
    )


@app.route('/hospital/login', methods=['GET', 'POST'])
def hospital_login():
    """Hospital login page."""
    if request.method == 'POST':
        hospital_id = request.form.get('hospital_id')
        api_key = request.form.get('api_key')

        # Simple validation (in production, use proper auth)
        hospital = get_hospital(hospital_id)
        if hospital and hospital.get('api_key') == api_key:
            session['hospital_id'] = hospital_id
            session['hospital_name'] = hospital['name']
            return redirect(url_for('hospital_dashboard'))
        else:
            return render_template('hospital/login.html', error='Invalid credentials')

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
        return redirect(url_for('hospital_login'))

    if request.method == 'POST':
        if 'images' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('images')
        hospital_id = session['hospital_id']

        # Create upload record
        upload = create_upload(
            hospital_id=hospital_id,
            user_id=session.get('user_id', 'doctor-001'),
            image_count=len(files)
        )

        results = []
        max_mb = Config.MAX_FILE_SIZE_MB
        for file in files:
            if not file or not file.filename:
                results.append({'filename': '(no filename)', 'status': 'skipped', 'error': 'Empty or missing file.'})
                continue
            if not allowed_file(file.filename):
                results.append({'filename': file.filename, 'status': 'skipped', 'error': 'File type not allowed. Use JPG or PNG.'})
                continue
            if not validate_file_size(file):
                results.append({'filename': file.filename, 'status': 'skipped', 'error': f'File too large. Maximum size is {max_mb} MB.'})
                continue
            try:
                # Get prediction from model API
                prediction = get_prediction(file)

                # Save analysis to database
                analysis = create_analysis(
                    upload_id=upload['id'],
                    image_path=f"uploads/{generate_unique_filename(file.filename)}",
                    prediction=prediction.get('prediction', 'UNCERTAIN'),
                    confidence=prediction.get('confidence', 0),
                    severity=get_severity_from_confidence(prediction.get('confidence', 0)),
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

                results.append({
                    'filename': file.filename,
                    'status': 'success',
                    'prediction': analysis['ai_prediction'],
                    'confidence': analysis['confidence']
                })

            except ModelAPIError as e:
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'error': str(e)
                })

        return jsonify({
            'upload_id': upload['id'],
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

    # Get global stats
    global_stats = get_global_stats(days=1)

    # Get regional data for map
    regional_data = get_regional_data(region_type='country')

    # Get active alerts
    alerts = get_active_alerts()

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
    supabase = get_supabase_client()

    response = supabase.table('alerts') \
        .select('*') \
        .order('triggered_at', desc=True) \
        .limit(100) \
        .execute()

    alerts = response.data

    return render_template('surveillance/alerts.html', alerts=alerts)


@app.route('/surveillance/hospitals')
def surveillance_hospitals():
    """View all participating hospitals and their status."""
    hospitals = get_all_hospitals()

    # Get stats for each hospital
    hospital_data = []
    for hospital in hospitals:
        stats = get_hospital_stats(hospital['id'], days=1)
        hospital['current_cases'] = stats['case_count']
        hospital['severe_cases'] = stats['severe_count']
        hospital_data.append(hospital)

    return render_template('surveillance/hospitals.html', hospitals=hospital_data)


# ===================== API ENDPOINTS =====================

@app.route('/api/v1/global-stats')
def api_global_stats():
    """API endpoint for global statistics."""
    days = request.args.get('days', default=1, type=int)
    stats = get_global_stats(days=days)
    return jsonify(stats)


@app.route('/api/v1/regional-data')
def api_regional_data():
    """API endpoint for regional data (for map)."""
    region_type = request.args.get('type', default='country')
    data = get_regional_data(region_type=region_type)
    return jsonify(data)


@app.route('/api/v1/alerts')
def api_alerts():
    """API endpoint for alerts."""
    alerts = get_active_alerts()
    return jsonify(alerts)


@app.route('/api/v1/hospital/<hospital_id>/stats')
def api_hospital_stats(hospital_id):
    """API endpoint for hospital-specific statistics."""
    days = request.args.get('days', default=1, type=int)
    stats = get_hospital_stats(hospital_id, days=days)
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
