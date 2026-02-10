"""
Predictive Analytics Module
Handles time-series forecasting, resource demand prediction, and trend analysis.

AI Attribution: This file was developed with assistance from Claude (Anthropic).
https://claude.ai
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Using fallback linear regression.")

from sklearn.linear_model import LinearRegression


class CaseForecastModel:
    """Time-series forecasting for case predictions."""

    def __init__(self, use_prophet: bool = True):
        """Initialize forecasting model.

        Args:
            use_prophet: Use Prophet if available, else use linear regression
        """
        self.use_prophet = use_prophet and PROPHET_AVAILABLE
        self.model = None

    def prepare_data(self, timeseries_data: List[Dict]) -> pd.DataFrame:
        """Convert database time-series to DataFrame.

        Args:
            timeseries_data: List of dicts with 'date' and case count fields

        Returns:
            DataFrame with 'ds' (date) and 'y' (value) columns for Prophet
        """
        if not timeseries_data:
            return pd.DataFrame(columns=['ds', 'y'])

        df = pd.DataFrame(timeseries_data)

        # Convert to Prophet format
        if 'date' in df.columns and 'case_count' in df.columns:
            df['ds'] = pd.to_datetime(df['date'])
            df['y'] = df['case_count']

        return df[['ds', 'y']].sort_values('ds')

    def fit(self, timeseries_data: List[Dict]) -> bool:
        """Train the forecasting model.

        Args:
            timeseries_data: Historical data with dates and case counts

        Returns:
            True if training successful, False otherwise
        """
        df = self.prepare_data(timeseries_data)

        if len(df) < 3:
            print("Warning: Insufficient data for forecasting (need at least 3 data points)")
            return False

        try:
            if self.use_prophet:
                # Use Prophet for more sophisticated forecasting
                self.model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    changepoint_prior_scale=0.05,  # Detect trend changes
                    interval_width=0.95  # 95% confidence intervals
                )
                self.model.fit(df)
            else:
                # Fallback: Linear regression
                df['day_num'] = (df['ds'] - df['ds'].min()).dt.days
                X = df[['day_num']].values
                y = df['y'].values

                self.model = LinearRegression()
                self.model.fit(X, y)
                self.base_date = df['ds'].min()

            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False

    def forecast(self, days: int = 7) -> Dict:
        """Generate forecast for next N days.

        Args:
            days: Number of days to forecast

        Returns:
            Dict with predictions, upper/lower bounds, and metadata
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not trained. Call fit() first.',
                'predictions': []
            }

        try:
            if self.use_prophet:
                # Prophet forecast
                future = self.model.make_future_dataframe(periods=days)
                forecast = self.model.predict(future)

                # Get only future predictions
                future_forecast = forecast.tail(days)

                predictions = []
                for _, row in future_forecast.iterrows():
                    predictions.append({
                        'date': row['ds'].strftime('%Y-%m-%d'),
                        'predicted_cases': max(0, int(round(row['yhat']))),
                        'lower_bound': max(0, int(round(row['yhat_lower']))),
                        'upper_bound': max(0, int(round(row['yhat_upper']))),
                        'confidence_interval': f"±{int(round((row['yhat_upper'] - row['yhat_lower']) / 2))}"
                    })

                return {
                    'success': True,
                    'predictions': predictions,
                    'model_type': 'prophet'
                }
            else:
                # Linear regression forecast
                predictions = []
                for day_offset in range(1, days + 1):
                    future_date = self.base_date + timedelta(days=day_offset)
                    day_num = (future_date - self.base_date).days

                    pred = self.model.predict([[day_num]])[0]

                    # Estimate confidence interval (±20% for simple linear model)
                    margin = pred * 0.2

                    predictions.append({
                        'date': future_date.strftime('%Y-%m-%d'),
                        'predicted_cases': max(0, int(round(pred))),
                        'lower_bound': max(0, int(round(pred - margin))),
                        'upper_bound': max(0, int(round(pred + margin))),
                        'confidence_interval': f"±{int(round(margin))}"
                    })

                return {
                    'success': True,
                    'predictions': predictions,
                    'model_type': 'linear_regression'
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }


class ResourceDemandPredictor:
    """Predict hospital resource needs based on case forecasts."""

    # Default conversion ratios (can be customized based on historical data)
    SEVERE_RATIO = 0.20  # 20% of pneumonia cases are severe
    ICU_BED_RATIO = 0.30  # 30% of pneumonia cases need ICU
    VENTILATOR_RATIO = 0.10  # 10% of pneumonia cases need ventilators
    OXYGEN_PER_CASE = 2  # Units of oxygen per pneumonia case per day
    AVG_STAY_DAYS = 7  # Average hospital stay for pneumonia patients

    def __init__(self, custom_ratios: Optional[Dict] = None):
        """Initialize resource predictor.

        Args:
            custom_ratios: Optional dict to override default ratios
        """
        if custom_ratios:
            self.SEVERE_RATIO = custom_ratios.get('severe_ratio', self.SEVERE_RATIO)
            self.ICU_BED_RATIO = custom_ratios.get('icu_bed_ratio', self.ICU_BED_RATIO)
            self.VENTILATOR_RATIO = custom_ratios.get('ventilator_ratio', self.VENTILATOR_RATIO)
            self.OXYGEN_PER_CASE = custom_ratios.get('oxygen_per_case', self.OXYGEN_PER_CASE)

    def predict_resource_needs(self, case_forecast: List[Dict],
                               current_capacity: Dict,
                               current_occupancy: Optional[Dict] = None) -> Dict:
        """Calculate resource requirements from case forecast.

        Args:
            case_forecast: List of daily predictions from CaseForecastModel
            current_capacity: Dict with total_beds, icu_beds
            current_occupancy: Optional dict with current beds_occupied, icu_beds_occupied

        Returns:
            Dict with daily resource needs and gap analysis
        """
        if not case_forecast:
            return {'success': False, 'error': 'No forecast data provided'}

        resource_timeline = []

        for day_pred in case_forecast:
            predicted_cases = day_pred['predicted_cases']

            # Calculate resource needs
            severe_cases = int(predicted_cases * self.SEVERE_RATIO)
            icu_beds_needed = int(predicted_cases * self.ICU_BED_RATIO)
            ventilators_needed = int(predicted_cases * self.VENTILATOR_RATIO)
            oxygen_needed = int(predicted_cases * self.OXYGEN_PER_CASE)

            # Calculate gaps
            icu_gap = max(0, icu_beds_needed - current_capacity.get('icu_beds', 0))

            resource_timeline.append({
                'date': day_pred['date'],
                'predicted_cases': predicted_cases,
                'severe_cases': severe_cases,
                'icu_beds_needed': icu_beds_needed,
                'ventilators_needed': ventilators_needed,
                'oxygen_units_needed': oxygen_needed,
                'icu_bed_gap': icu_gap,
                'icu_capacity_utilization': round(
                    (icu_beds_needed / current_capacity.get('icu_beds', 1)) * 100, 1
                )
            })

        # Generate summary
        peak_day = max(resource_timeline, key=lambda x: x['predicted_cases'])
        total_icu_gap = sum(day['icu_bed_gap'] for day in resource_timeline)

        return {
            'success': True,
            'timeline': resource_timeline,
            'summary': {
                'peak_date': peak_day['date'],
                'peak_cases': peak_day['predicted_cases'],
                'peak_icu_beds_needed': peak_day['icu_beds_needed'],
                'peak_ventilators_needed': peak_day['ventilators_needed'],
                'total_icu_gap_days': total_icu_gap,
                'max_icu_utilization': max(day['icu_capacity_utilization']
                                          for day in resource_timeline)
            }
        }


class GrowthAnalyzer:
    """Analyze epidemic growth patterns and detect rapid acceleration."""

    @staticmethod
    def calculate_growth_metrics(timeseries_data: List[Dict]) -> Dict:
        """Calculate growth rate, doubling time, and velocity.

        Args:
            timeseries_data: Historical data with dates and case counts

        Returns:
            Dict with growth metrics
        """
        if len(timeseries_data) < 2:
            return {
                'success': False,
                'error': 'Insufficient data (need at least 2 days)'
            }

        df = pd.DataFrame(timeseries_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Calculate daily changes
        df['daily_change'] = df['case_count'].diff()
        df['percent_change'] = df['case_count'].pct_change() * 100

        # Recent metrics (last 3 days vs previous 3 days)
        recent_cases = df['case_count'].tail(3).sum()
        previous_cases = df['case_count'].tail(6).head(3).sum()

        if previous_cases > 0:
            growth_rate = ((recent_cases - previous_cases) / previous_cases) * 100
        else:
            growth_rate = 0

        # Doubling time (using last 7 days)
        last_week = df.tail(7)
        if len(last_week) >= 2:
            first_val = last_week['case_count'].iloc[0]
            last_val = last_week['case_count'].iloc[-1]

            if first_val > 0 and last_val > first_val:
                days_elapsed = len(last_week) - 1
                growth_factor = last_val / first_val
                doubling_time = (days_elapsed * np.log(2)) / np.log(growth_factor)
            else:
                doubling_time = None
        else:
            doubling_time = None

        # Velocity (average daily increase in last 7 days)
        velocity = df['daily_change'].tail(7).mean()

        # Trend direction
        if growth_rate > 20:
            trend = 'rapid_growth'
        elif growth_rate > 5:
            trend = 'growing'
        elif growth_rate > -5:
            trend = 'stable'
        else:
            trend = 'declining'

        return {
            'success': True,
            'growth_rate_3day': round(growth_rate, 1),
            'doubling_time_days': round(doubling_time, 1) if doubling_time else None,
            'daily_velocity': round(velocity, 1) if not np.isnan(velocity) else 0,
            'trend': trend,
            'current_cases': int(df['case_count'].iloc[-1]),
            'previous_cases': int(df['case_count'].iloc[0]) if len(df) > 0 else 0,
            'latest_date': df['date'].iloc[-1].strftime('%Y-%m-%d')
        }

    @staticmethod
    def detect_surge(timeseries_data: List[Dict], threshold: float = 50.0) -> Dict:
        """Detect if region is experiencing a case surge.

        Args:
            timeseries_data: Historical case data
            threshold: Growth rate % to trigger surge alert

        Returns:
            Dict with surge detection results
        """
        metrics = GrowthAnalyzer.calculate_growth_metrics(timeseries_data)

        if not metrics['success']:
            return metrics

        is_surge = metrics['growth_rate_3day'] > threshold

        # Determine severity
        if metrics['growth_rate_3day'] > 100:
            severity = 'critical'
        elif metrics['growth_rate_3day'] > 50:
            severity = 'high'
        elif metrics['growth_rate_3day'] > 20:
            severity = 'medium'
        else:
            severity = 'low'

        return {
            'success': True,
            'is_surge': is_surge,
            'severity': severity,
            'metrics': metrics,
            'alert_message': f"Cases increased by {metrics['growth_rate_3day']}% in 3 days" if is_surge else None
        }


def generate_forecast_report(region_name: str, timeseries_data: List[Dict],
                             current_capacity: Dict, forecast_days: int = 7) -> Dict:
    """Generate complete forecast report for a region.

    Args:
        region_name: Name of region
        timeseries_data: Historical case data
        current_capacity: Hospital capacity info
        forecast_days: Days to forecast

    Returns:
        Complete forecast report with cases, resources, and alerts
    """
    # Case forecast
    forecast_model = CaseForecastModel()
    if not forecast_model.fit(timeseries_data):
        return {
            'success': False,
            'error': 'Failed to train forecast model'
        }

    case_forecast = forecast_model.forecast(days=forecast_days)

    if not case_forecast['success']:
        return case_forecast

    # Resource predictions
    resource_predictor = ResourceDemandPredictor()
    resource_forecast = resource_predictor.predict_resource_needs(
        case_forecast['predictions'],
        current_capacity
    )

    # Growth analysis
    growth_analysis = GrowthAnalyzer.calculate_growth_metrics(timeseries_data)
    surge_detection = GrowthAnalyzer.detect_surge(timeseries_data)

    return {
        'success': True,
        'region_name': region_name,
        'forecast_generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'case_forecast': case_forecast['predictions'],
        'resource_forecast': resource_forecast,
        'growth_analysis': growth_analysis,
        'surge_detection': surge_detection,
        'recommendations': generate_recommendations(
            surge_detection,
            resource_forecast,
            growth_analysis
        )
    }


def generate_recommendations(surge_info: Dict, resource_forecast: Dict,
                            growth_metrics: Dict) -> List[str]:
    """Generate actionable recommendations for policymakers.

    Args:
        surge_info: Surge detection results
        resource_forecast: Resource demand predictions
        growth_metrics: Growth analysis metrics

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Surge recommendations
    if surge_info.get('is_surge'):
        if surge_info['severity'] == 'critical':
            recommendations.append(
                "🔴 CRITICAL: Implement immediate lockdown measures and expand testing"
            )
        elif surge_info['severity'] == 'high':
            recommendations.append(
                "🟠 HIGH ALERT: Increase hospital preparedness and public health messaging"
            )

    # Resource recommendations
    if resource_forecast.get('success'):
        summary = resource_forecast.get('summary', {})
        max_util = summary.get('max_icu_utilization', 0)

        if max_util > 100:
            gap = summary.get('peak_icu_beds_needed', 0) - current_capacity.get('icu_beds', 0)
            recommendations.append(
                f"🏥 URGENT: Procure {gap} additional ICU beds before {summary.get('peak_date')}"
            )
        elif max_util > 80:
            recommendations.append(
                "⚠️ Prepare surge capacity - ICU utilization will exceed 80%"
            )

        # Ventilator needs
        peak_vents = summary.get('peak_ventilators_needed', 0)
        if peak_vents > 0:
            recommendations.append(
                f"💨 Ensure availability of {peak_vents} ventilators by {summary.get('peak_date')}"
            )

    # Doubling time recommendations
    if growth_metrics.get('doubling_time_days'):
        doubling = growth_metrics['doubling_time_days']
        if doubling < 5:
            recommendations.append(
                f"⏱️ Cases doubling every {doubling:.1f} days - Exponential growth in progress"
            )

    # General recommendations
    if growth_metrics.get('trend') == 'rapid_growth':
        recommendations.append(
            "📊 Prioritize vulnerable populations (60+) for vaccination and monitoring"
        )
        recommendations.append(
            "🧪 Scale up testing capacity to match case growth trajectory"
        )

    return recommendations if recommendations else ["✅ Situation stable - Continue monitoring"]
