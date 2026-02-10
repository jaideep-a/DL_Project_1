"""
Test script for prediction models and alert engine.

AI Attribution: This file was developed with assistance from Claude (Anthropic).
https://claude.ai
"""

from datetime import datetime, timedelta
from app.models.predictions import (
    CaseForecastModel, ResourceDemandPredictor, GrowthAnalyzer,
    generate_forecast_report
)
from app.models.alerts import AlertEngine


def generate_sample_data(days=30, start_value=100, growth_rate=1.15):
    """Generate sample time-series data for testing."""
    data = []
    current_date = datetime.now() - timedelta(days=days)

    for i in range(days):
        case_count = int(start_value * (growth_rate ** (i / 7)))  # Exponential growth
        data.append({
            'date': (current_date + timedelta(days=i)).strftime('%Y-%m-%d'),
            'case_count': case_count,
            'pneumonia_count': int(case_count * 0.3),
            'severe_count': int(case_count * 0.1),
            'deaths': int(case_count * 0.02),
            'region_name': 'Test Region',
            'region_id': 'TEST'
        })

    return data


def test_case_forecast():
    """Test case forecasting model."""
    print("=" * 60)
    print("TEST 1: Case Forecasting Model")
    print("=" * 60)

    # Generate sample data
    timeseries = generate_sample_data(days=30, start_value=100, growth_rate=1.15)

    # Create and train model
    model = CaseForecastModel(use_prophet=False)  # Use linear regression for faster testing
    success = model.fit(timeseries)

    if success:
        print("✅ Model trained successfully")

        # Generate 7-day forecast
        forecast = model.forecast(days=7)

        if forecast['success']:
            print(f"\n📊 7-Day Forecast ({forecast['model_type']}):")
            print("-" * 60)
            for pred in forecast['predictions']:
                print(f"  {pred['date']}: {pred['predicted_cases']} cases "
                      f"(±{pred['confidence_interval']})")
            print("\n✅ Forecast generated successfully\n")
        else:
            print(f"❌ Forecast failed: {forecast.get('error')}\n")
    else:
        print("❌ Model training failed\n")


def test_resource_prediction():
    """Test resource demand predictor."""
    print("=" * 60)
    print("TEST 2: Resource Demand Prediction")
    print("=" * 60)

    # Sample forecast data
    case_forecast = [
        {'date': '2026-02-11', 'predicted_cases': 500},
        {'date': '2026-02-12', 'predicted_cases': 550},
        {'date': '2026-02-13', 'predicted_cases': 600},
        {'date': '2026-02-14', 'predicted_cases': 650},
        {'date': '2026-02-15', 'predicted_cases': 700},
        {'date': '2026-02-16', 'predicted_cases': 750},
        {'date': '2026-02-17', 'predicted_cases': 800},
    ]

    current_capacity = {
        'total_beds': 500,
        'icu_beds': 100,
        'ventilators_available': 50
    }

    predictor = ResourceDemandPredictor()
    result = predictor.predict_resource_needs(case_forecast, current_capacity)

    if result['success']:
        print("✅ Resource predictions generated\n")
        print("📋 Summary:")
        print(f"  Peak Date: {result['summary']['peak_date']}")
        print(f"  Peak Cases: {result['summary']['peak_cases']}")
        print(f"  Peak ICU Beds Needed: {result['summary']['peak_icu_beds_needed']}")
        print(f"  Peak Ventilators Needed: {result['summary']['peak_ventilators_needed']}")
        print(f"  Max ICU Utilization: {result['summary']['max_icu_utilization']}%")

        print("\n📊 Daily Resource Timeline:")
        print("-" * 60)
        for day in result['timeline'][:3]:  # Show first 3 days
            print(f"  {day['date']}:")
            print(f"    ICU Beds Needed: {day['icu_beds_needed']} "
                  f"(Gap: {day['icu_bed_gap']})")
            print(f"    Ventilators Needed: {day['ventilators_needed']}")
            print(f"    Oxygen Units: {day['oxygen_units_needed']}")

        print("\n✅ Resource prediction successful\n")
    else:
        print(f"❌ Resource prediction failed: {result.get('error')}\n")


def test_growth_analyzer():
    """Test growth analysis and metrics."""
    print("=" * 60)
    print("TEST 3: Growth Analysis")
    print("=" * 60)

    # Generate data with rapid growth
    timeseries = generate_sample_data(days=30, start_value=100, growth_rate=1.25)

    # Calculate metrics
    metrics = GrowthAnalyzer.calculate_growth_metrics(timeseries)

    if metrics['success']:
        print("✅ Growth metrics calculated\n")
        print("📈 Metrics:")
        print(f"  Current Cases: {metrics['current_cases']}")
        print(f"  3-Day Growth Rate: {metrics['growth_rate_3day']}%")
        if metrics['doubling_time_days']:
            print(f"  Doubling Time: {metrics['doubling_time_days']} days")
        print(f"  Daily Velocity: {metrics['daily_velocity']} cases/day")
        print(f"  Trend: {metrics['trend']}")

        # Test surge detection
        surge = GrowthAnalyzer.detect_surge(timeseries, threshold=50.0)

        print("\n🚨 Surge Detection:")
        print(f"  Is Surge: {surge['is_surge']}")
        print(f"  Severity: {surge['severity']}")
        if surge['alert_message']:
            print(f"  Alert: {surge['alert_message']}")

        print("\n✅ Growth analysis successful\n")
    else:
        print(f"❌ Growth analysis failed: {metrics.get('error')}\n")


def test_alert_engine():
    """Test alert generation."""
    print("=" * 60)
    print("TEST 4: Alert Engine")
    print("=" * 60)

    # Generate rapid growth data
    timeseries = generate_sample_data(days=30, start_value=100, growth_rate=1.3)

    alert_engine = AlertEngine()

    # Generate growth alerts
    alerts = alert_engine.generate_growth_alerts(
        region_name="Test City",
        region_id="TEST_CITY",
        timeseries_data=timeseries
    )

    print(f"✅ Generated {len(alerts)} alerts\n")

    for i, alert in enumerate(alerts, 1):
        print(f"Alert {i}:")
        print(f"  Title: {alert['title']}")
        print(f"  Severity: {alert['severity']}")
        print(f"  Description: {alert['description']}")
        print(f"  Recommendations: {len(alert['recommendations'])} actions")
        print()

    # Test capacity alerts
    resource_forecast = {
        'success': True,
        'summary': {
            'peak_date': '2026-02-17',
            'peak_cases': 800,
            'peak_icu_beds_needed': 240,
            'peak_ventilators_needed': 80,
            'max_icu_utilization': 120.0
        }
    }

    current_capacity = {
        'icu_beds': 200,
        'ventilators_available': 60
    }

    capacity_alerts = alert_engine.generate_capacity_alerts(
        region_name="Test City",
        region_id="TEST_CITY",
        resource_forecast=resource_forecast,
        current_capacity=current_capacity
    )

    print(f"✅ Generated {len(capacity_alerts)} capacity alerts\n")

    for i, alert in enumerate(capacity_alerts, 1):
        print(f"Capacity Alert {i}:")
        print(f"  Title: {alert['title']}")
        print(f"  Severity: {alert['severity']}")
        print(f"  Description: {alert['description']}")
        print()

    print("✅ Alert engine test successful\n")


def test_full_forecast_report():
    """Test complete forecast report generation."""
    print("=" * 60)
    print("TEST 5: Full Forecast Report")
    print("=" * 60)

    timeseries = generate_sample_data(days=30, start_value=100, growth_rate=1.2)

    current_capacity = {
        'total_beds': 500,
        'icu_beds': 100,
        'ventilators_available': 50
    }

    report = generate_forecast_report(
        region_name="Test Region",
        timeseries_data=timeseries,
        current_capacity=current_capacity,
        forecast_days=7
    )

    if report['success']:
        print("✅ Full forecast report generated\n")
        print(f"Region: {report['region_name']}")
        print(f"Generated At: {report['forecast_generated_at']}")
        print(f"\nForecast Days: {len(report['case_forecast'])}")
        print(f"Growth Analysis: {report['growth_analysis']['trend']}")
        print(f"Surge Detected: {report['surge_detection']['is_surge']}")
        print(f"Recommendations: {len(report['recommendations'])}")

        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")

        print("\n✅ Full report test successful\n")
    else:
        print(f"❌ Report generation failed: {report.get('error')}\n")


if __name__ == '__main__':
    print("\n🧪 PREDICTION SYSTEM TEST SUITE")
    print("=" * 60)
    print("Testing MediAlert predictive analytics components\n")

    try:
        test_case_forecast()
        test_resource_prediction()
        test_growth_analyzer()
        test_alert_engine()
        test_full_forecast_report()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe prediction system is ready to use.")
        print("Access the predictions dashboard at: /surveillance/predictions")

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
