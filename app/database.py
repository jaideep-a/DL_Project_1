# AI Attribution: This file was developed with assistance from Claude (Anthropic).
# https://claude.ai

import os
from supabase import create_client, Client
from .config import Config

# Initialize Supabase client
def get_supabase_client() -> Client:
    """Get or create Supabase client."""
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')

    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")

    return create_client(supabase_url, supabase_key)


# Database helper functions
def create_hospital(name: str, city: str, state: str, country: str,
                   latitude: float, longitude: float, registration_number: str,
                   total_beds: int, icu_beds: int) -> dict:
    """Create a new hospital record."""
    supabase = get_supabase_client()
    data = {
        'name': name,
        'city': city,
        'state': state,
        'country': country,
        'latitude': latitude,
        'longitude': longitude,
        'registration_number': registration_number,
        'total_beds': total_beds,
        'icu_beds': icu_beds,
    }
    response = supabase.table('hospitals').insert(data).execute()
    return response.data[0] if response.data else None


def get_hospital(hospital_id: str) -> dict:
    """Get hospital by ID."""
    supabase = get_supabase_client()
    response = supabase.table('hospitals').select('*').eq('id', hospital_id).execute()
    return response.data[0] if response.data else None


def get_all_hospitals() -> list:
    """Get all hospitals."""
    supabase = get_supabase_client()
    response = supabase.table('hospitals').select('*').execute()
    return response.data


def create_upload(hospital_id: str, user_id: str, image_count: int) -> dict:
    """Create a new upload record."""
    supabase = get_supabase_client()
    data = {
        'hospital_id': hospital_id,
        'user_id': user_id,
        'image_count': image_count,
        'status': 'processing',
    }
    response = supabase.table('uploads').insert(data).execute()
    return response.data[0] if response.data else None


def create_analysis(upload_id: str, image_path: str, prediction: str,
                   confidence: float, severity: str, processing_time_ms: int,
                   model_version: str, heatmap_path: str = None) -> dict:
    """Create a new analysis record."""
    supabase = get_supabase_client()
    data = {
        'upload_id': upload_id,
        'image_path': image_path,
        'ai_prediction': prediction,
        'confidence': confidence,
        'severity': severity,
        'processing_time_ms': processing_time_ms,
        'model_version': model_version,
        'heatmap_path': heatmap_path,
    }
    response = supabase.table('analyses').insert(data).execute()
    return response.data[0] if response.data else None


def create_patient_metadata(analysis_id: str, age_range: str, gender: str,
                           vaccination_status: str, symptoms: list, outcome: str = 'unknown') -> dict:
    """Create patient metadata record."""
    supabase = get_supabase_client()
    data = {
        'analysis_id': analysis_id,
        'age_range': age_range,
        'gender': gender,
        'vaccination_status': vaccination_status,
        'symptoms': symptoms,
        'outcome': outcome,
    }
    response = supabase.table('patient_metadata').insert(data).execute()
    return response.data[0] if response.data else None


def get_hospital_stats(hospital_id: str, days: int = 1) -> dict:
    """Get hospital statistics for the past N days."""
    supabase = get_supabase_client()

    # Get today's summary
    response = supabase.table('case_summary') \
        .select('*') \
        .eq('hospital_id', hospital_id) \
        .gte('date', f'now - interval \'{days} days\'') \
        .execute()

    if not response.data:
        return {
            'case_count': 0,
            'normal_count': 0,
            'pneumonia_count': 0,
            'severe_count': 0,
            'deaths': 0,
            'avg_confidence': 0,
        }

    # Aggregate
    total = {
        'case_count': sum(r['case_count'] for r in response.data),
        'normal_count': sum(r['normal_count'] for r in response.data),
        'pneumonia_count': sum(r['pneumonia_count'] for r in response.data),
        'severe_count': sum(r['severe_count'] for r in response.data),
        'deaths': sum(r['deaths'] for r in response.data),
        'avg_confidence': sum(r['avg_confidence'] for r in response.data) / len(response.data),
    }
    return total


def get_global_stats(days: int = 1) -> dict:
    """Get global statistics for the past N days."""
    supabase = get_supabase_client()

    response = supabase.table('regional_summary') \
        .select('*') \
        .eq('region_type', 'country') \
        .gte('date', f'now - interval \'{days} days\'') \
        .execute()

    if not response.data:
        return {
            'case_count': 0,
            'normal_count': 0,
            'pneumonia_count': 0,
            'severe_count': 0,
            'deaths': 0,
        }

    # Aggregate
    total = {
        'case_count': sum(r['case_count'] for r in response.data),
        'normal_count': sum(r['normal_count'] for r in response.data),
        'pneumonia_count': sum(r['pneumonia_count'] for r in response.data),
        'severe_count': sum(r['severe_count'] for r in response.data),
        'deaths': sum(r['deaths'] for r in response.data),
    }
    return total


def get_regional_data(region_type: str = 'country') -> list:
    """Get regional data for map visualization."""
    supabase = get_supabase_client()

    response = supabase.table('regional_summary') \
        .select('*') \
        .eq('region_type', region_type) \
        .order('case_count', desc=True) \
        .execute()

    return response.data


def create_alert(region_id: str, alert_type: str, severity: str, description: str, recipients: list) -> dict:
    """Create an alert."""
    supabase = get_supabase_client()
    data = {
        'region_id': region_id,
        'alert_type': alert_type,
        'severity': severity,
        'description': description,
        'recipients': recipients,
    }
    response = supabase.table('alerts').insert(data).execute()
    return response.data[0] if response.data else None


def get_active_alerts() -> list:
    """Get all active (unresolved) alerts."""
    supabase = get_supabase_client()
    response = supabase.table('alerts') \
        .select('*') \
        .is_('resolved_at', True) \
        .order('triggered_at', desc=True) \
        .execute()

    return response.data


# Time-series data functions for predictions
def get_regional_timeseries(region_id: str = None, region_type: str = 'country', days: int = 30) -> list:
    """Get time-series data for a region (for forecasting)."""
    supabase = get_supabase_client()

    query = supabase.table('regional_summary') \
        .select('date, case_count, pneumonia_count, severe_count, deaths, region_name, region_id') \
        .eq('region_type', region_type) \
        .gte('date', f'now - interval \'{days} days\'') \
        .order('date', desc=False)

    if region_id:
        query = query.eq('region_id', region_id)

    response = query.execute()
    return response.data


def get_hospital_timeseries(hospital_id: str = None, days: int = 30) -> list:
    """Get time-series data for a hospital (for forecasting)."""
    supabase = get_supabase_client()

    query = supabase.table('case_summary') \
        .select('date, case_count, pneumonia_count, severe_count, deaths, hospital_id') \
        .gte('date', f'now - interval \'{days} days\'') \
        .order('date', desc=False)

    if hospital_id:
        query = query.eq('hospital_id', hospital_id)

    response = query.execute()
    return response.data


def get_resource_timeseries(hospital_id: str = None, days: int = 30) -> list:
    """Get resource availability time-series (beds, ventilators, oxygen)."""
    supabase = get_supabase_client()

    query = supabase.table('resources') \
        .select('date, hospital_id, icu_beds_available, ventilators_available, oxygen_supply_days, staff_available') \
        .gte('date', f'now - interval \'{days} days\'') \
        .order('date', desc=False)

    if hospital_id:
        query = query.eq('hospital_id', hospital_id)

    response = query.execute()
    return response.data


def get_current_hospital_capacity(hospital_id: str = None) -> list:
    """Get current hospital capacity (total beds, ICU beds) and latest resource availability."""
    supabase = get_supabase_client()

    # Get hospital capacity
    query = supabase.table('hospitals') \
        .select('id, name, city, state, country, total_beds, icu_beds')

    if hospital_id:
        query = query.eq('id', hospital_id)

    hospitals = query.execute().data

    # Get latest resource data for each hospital
    for hospital in hospitals:
        resource_query = supabase.table('resources') \
            .select('icu_beds_available, ventilators_available, oxygen_supply_days, staff_available, date') \
            .eq('hospital_id', hospital['id']) \
            .order('date', desc=True) \
            .limit(1)

        resource_data = resource_query.execute().data
        if resource_data:
            hospital['latest_resources'] = resource_data[0]
        else:
            hospital['latest_resources'] = None

    return hospitals


def get_regional_summary_latest(region_type: str = 'country') -> list:
    """Get latest regional summary data."""
    supabase = get_supabase_client()

    # Get the most recent date first
    date_response = supabase.table('regional_summary') \
        .select('date') \
        .eq('region_type', region_type) \
        .order('date', desc=True) \
        .limit(1) \
        .execute()

    if not date_response.data:
        return []

    latest_date = date_response.data[0]['date']

    # Get all regions for that date
    response = supabase.table('regional_summary') \
        .select('*') \
        .eq('region_type', region_type) \
        .eq('date', latest_date) \
        .order('case_count', desc=True) \
        .execute()

    return response.data
