import os
<<<<<<< HEAD
from datetime import datetime
=======
import base64
from datetime import datetime, timedelta
>>>>>>> cb60b733f9c7b2a5eab98f077537fca3a0e470cd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from .config import Config
from .api_client import get_prediction, check_model_health, ModelAPIError
from .utils import allowed_file, validate_file_size, generate_unique_filename
from .database import (
    get_supabase_client, create_hospital, get_hospital, get_all_hospitals,
    create_upload, create_analysis, create_patient_metadata,
    get_hospital_stats, get_global_stats, get_regional_data, create_alert,
    get_active_alerts, get_regional_timeseries, get_hospital_timeseries,
    get_resource_timeseries, get_current_hospital_capacity, get_regional_summary_latest
)
from .models.predictions import (
    CaseForecastModel, ResourceDemandPredictor, GrowthAnalyzer,
    generate_forecast_report
)
from .models.alerts import AlertEngine


app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')
app.secret_key = Config.SECRET_KEY
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=Config.SESSION_LIFETIME_HOURS)
app.config['SESSION_COOKIE_SECURE'] = not Config.DEBUG  # Use secure cookies in production
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


@app.route('/surveillance/predictions')
def surveillance_predictions():
    """Policymaker predictions dashboard - forecasts and alerts."""
    # Get growth metrics for top regions
    regions = get_regional_summary_latest(region_type='country')

    growth_metrics = []
    rapid_growth_alerts = []

    alert_engine = AlertEngine()

    for region in regions[:10]:  # Top 10 regions
        region_id = region.get('region_id')
        region_name = region.get('region_name', region_id)

        # Get time-series data
        timeseries = get_regional_timeseries(
            region_id=region_id,
            region_type='country',
            days=30
        )

        if timeseries:
            # Calculate metrics
            metrics = GrowthAnalyzer.calculate_growth_metrics(timeseries)

            if metrics.get('success'):
                metrics['region_id'] = region_id
                metrics['region_name'] = region_name
                growth_metrics.append(metrics)

                # Generate growth alerts
                alerts = alert_engine.generate_growth_alerts(
                    region_name=region_name,
                    region_id=region_id,
                    timeseries_data=timeseries
                )
                rapid_growth_alerts.extend(alerts)

    # Sort metrics by growth rate
    growth_metrics.sort(key=lambda x: x.get('growth_rate_3day', 0), reverse=True)

    # Get alert summary
    alert_summary = alert_engine.get_alert_summary(rapid_growth_alerts)

    return render_template('surveillance/predictions.html',
        growth_metrics=growth_metrics,
        alerts=rapid_growth_alerts,
        alert_summary=alert_summary
    )


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


# ===================== PREDICTION & ALERT API ENDPOINTS =====================

@app.route('/api/v1/predictions/region/<region_id>')
def api_regional_predictions(region_id):
    """Generate 7-day forecast for a specific region.

    Query params:
        - region_type: 'country', 'state', or 'city' (default: 'country')
        - forecast_days: Number of days to forecast (default: 7)
    """
    try:
        region_type = request.args.get('region_type', default='country')
        forecast_days = request.args.get('forecast_days', default=7, type=int)

        # Get historical data (30 days)
        timeseries_data = get_regional_timeseries(
            region_id=region_id,
            region_type=region_type,
            days=30
        )

        if not timeseries_data:
            return jsonify({
                'success': False,
                'error': 'No historical data available for this region'
            }), 404

        region_name = timeseries_data[0].get('region_name', region_id)

        # Get current hospital capacity for this region
        hospitals = get_current_hospital_capacity()

        # Aggregate capacity across region
        total_capacity = {
            'total_beds': sum(h.get('total_beds', 0) for h in hospitals),
            'icu_beds': sum(h.get('icu_beds', 0) for h in hospitals),
            'ventilators_available': sum(
                h.get('latest_resources', {}).get('ventilators_available', 0)
                if h.get('latest_resources') else 0
                for h in hospitals
            )
        }

        # Generate comprehensive forecast report
        report = generate_forecast_report(
            region_name=region_name,
            timeseries_data=timeseries_data,
            current_capacity=total_capacity,
            forecast_days=forecast_days
        )

        return jsonify(report)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/predictions/hospital/<hospital_id>')
def api_hospital_predictions(hospital_id):
    """Generate 7-day forecast for a specific hospital.

    Query params:
        - forecast_days: Number of days to forecast (default: 7)
    """
    try:
        forecast_days = request.args.get('forecast_days', default=7, type=int)

        # Get hospital info
        hospital = get_hospital(hospital_id)
        if not hospital:
            return jsonify({
                'success': False,
                'error': 'Hospital not found'
            }), 404

        # Get historical data
        timeseries_data = get_hospital_timeseries(hospital_id=hospital_id, days=30)

        if not timeseries_data:
            return jsonify({
                'success': False,
                'error': 'No historical data available for this hospital'
            }), 404

        # Get hospital capacity
        capacity_info = get_current_hospital_capacity(hospital_id=hospital_id)
        current_capacity = capacity_info[0] if capacity_info else {}

        # Generate forecast report
        report = generate_forecast_report(
            region_name=hospital.get('name', hospital_id),
            timeseries_data=timeseries_data,
            current_capacity={
                'total_beds': current_capacity.get('total_beds', 0),
                'icu_beds': current_capacity.get('icu_beds', 0),
                'ventilators_available': (
                    current_capacity.get('latest_resources', {}).get('ventilators_available', 0)
                    if current_capacity.get('latest_resources') else 0
                )
            },
            forecast_days=forecast_days
        )

        return jsonify(report)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/alerts/growth')
def api_growth_alerts():
    """Get rapid growth alerts for all regions.

    Query params:
        - region_type: 'country', 'state', or 'city' (default: 'country')
        - threshold: Growth rate % to trigger alert (default: 50)
    """
    try:
        region_type = request.args.get('region_type', default='country')
        threshold = request.args.get('threshold', default=50.0, type=float)

        # Get latest regional data
        regions = get_regional_summary_latest(region_type=region_type)

        alert_engine = AlertEngine(thresholds={'surge_growth_rate': threshold})
        all_alerts = []

        for region in regions:
            region_id = region.get('region_id')
            region_name = region.get('region_name', region_id)

            # Get time-series for this region
            timeseries = get_regional_timeseries(
                region_id=region_id,
                region_type=region_type,
                days=30
            )

            if timeseries:
                # Generate growth alerts
                alerts = alert_engine.generate_growth_alerts(
                    region_name=region_name,
                    region_id=region_id,
                    timeseries_data=timeseries
                )
                all_alerts.extend(alerts)

        # Get summary
        summary = alert_engine.get_alert_summary(all_alerts)

        return jsonify({
            'success': True,
            'alerts': all_alerts,
            'summary': summary
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/alerts/capacity')
def api_capacity_alerts():
    """Get capacity warnings for all regions.

    Query params:
        - region_type: 'country', 'state', or 'city' (default: 'country')
        - forecast_days: Days to forecast (default: 7)
    """
    try:
        region_type = request.args.get('region_type', default='country')
        forecast_days = request.args.get('forecast_days', default=7, type=int)

        regions = get_regional_summary_latest(region_type=region_type)

        alert_engine = AlertEngine()
        all_alerts = []

        for region in regions[:10]:  # Limit to top 10 regions for performance
            region_id = region.get('region_id')
            region_name = region.get('region_name', region_id)

            # Get time-series
            timeseries = get_regional_timeseries(
                region_id=region_id,
                region_type=region_type,
                days=30
            )

            if not timeseries:
                continue

            # Get capacity
            hospitals = get_current_hospital_capacity()
            total_capacity = {
                'icu_beds': sum(h.get('icu_beds', 0) for h in hospitals),
                'ventilators_available': sum(
                    h.get('latest_resources', {}).get('ventilators_available', 0)
                    if h.get('latest_resources') else 0
                    for h in hospitals
                )
            }

            # Generate forecast
            forecast_model = CaseForecastModel()
            if forecast_model.fit(timeseries):
                case_forecast = forecast_model.forecast(days=forecast_days)

                if case_forecast.get('success'):
                    # Predict resource needs
                    resource_predictor = ResourceDemandPredictor()
                    resource_forecast = resource_predictor.predict_resource_needs(
                        case_forecast['predictions'],
                        total_capacity
                    )

                    # Generate capacity alerts
                    alerts = alert_engine.generate_capacity_alerts(
                        region_name=region_name,
                        region_id=region_id,
                        resource_forecast=resource_forecast,
                        current_capacity=total_capacity
                    )
                    all_alerts.extend(alerts)

        summary = alert_engine.get_alert_summary(all_alerts)

        return jsonify({
            'success': True,
            'alerts': all_alerts,
            'summary': summary
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/analytics/growth-metrics')
def api_growth_metrics():
    """Get growth metrics for all regions (doubling time, velocity, etc).

    Query params:
        - region_type: 'country', 'state', or 'city' (default: 'country')
    """
    try:
        region_type = request.args.get('region_type', default='country')

        regions = get_regional_summary_latest(region_type=region_type)

        metrics_list = []

        for region in regions:
            region_id = region.get('region_id')
            region_name = region.get('region_name', region_id)

            # Get time-series
            timeseries = get_regional_timeseries(
                region_id=region_id,
                region_type=region_type,
                days=30
            )

            if timeseries:
                # Calculate growth metrics
                metrics = GrowthAnalyzer.calculate_growth_metrics(timeseries)

                if metrics.get('success'):
                    metrics['region_id'] = region_id
                    metrics['region_name'] = region_name
                    metrics_list.append(metrics)

        # Sort by growth rate (descending)
        metrics_list.sort(key=lambda x: x.get('growth_rate_3day', 0), reverse=True)

        return jsonify({
            'success': True,
            'metrics': metrics_list,
            'total_regions': len(metrics_list)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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
