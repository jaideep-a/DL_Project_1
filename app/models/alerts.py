"""
Alert Engine Module
Generates automated alerts for rapid growth, capacity warnings, and resource depletion.

AI Attribution: This file was developed with assistance from Claude (Anthropic).
https://claude.ai
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .predictions import GrowthAnalyzer


class AlertEngine:
    """Generate automated alerts based on predictions and thresholds."""

    # Alert severity levels
    SEVERITY_CRITICAL = 'critical'
    SEVERITY_HIGH = 'high'
    SEVERITY_MEDIUM = 'medium'
    SEVERITY_LOW = 'low'

    # Alert types
    TYPE_SURGE = 'surge_detected'
    TYPE_CAPACITY = 'capacity_warning'
    TYPE_RESOURCE_DEPLETION = 'resource_depletion'
    TYPE_RAPID_GROWTH = 'rapid_growth'
    TYPE_DEMOGRAPHIC_RISK = 'demographic_risk'

    def __init__(self, thresholds: Optional[Dict] = None):
        """Initialize alert engine.

        Args:
            thresholds: Optional dict to customize alert thresholds
        """
        # Default thresholds
        self.thresholds = {
            'surge_growth_rate': 50.0,  # % growth in 3 days
            'rapid_growth_rate': 20.0,
            'doubling_time_critical': 3.0,  # days
            'doubling_time_warning': 7.0,
            'icu_utilization_critical': 90.0,  # % of capacity
            'icu_utilization_warning': 80.0,
            'oxygen_days_critical': 3,  # days remaining
            'oxygen_days_warning': 7,
        }

        if thresholds:
            self.thresholds.update(thresholds)

    def generate_growth_alerts(self, region_name: str, region_id: str,
                               timeseries_data: List[Dict]) -> List[Dict]:
        """Generate alerts for rapid case growth.

        Args:
            region_name: Name of region
            region_id: Region identifier
            timeseries_data: Historical case data

        Returns:
            List of alert dicts
        """
        alerts = []

        # Calculate growth metrics
        growth_metrics = GrowthAnalyzer.calculate_growth_metrics(timeseries_data)

        if not growth_metrics.get('success'):
            return alerts

        growth_rate = growth_metrics['growth_rate_3day']
        doubling_time = growth_metrics.get('doubling_time_days')
        trend = growth_metrics['trend']

        # Surge alert
        if growth_rate >= self.thresholds['surge_growth_rate']:
            alerts.append({
                'region_id': region_id,
                'region_name': region_name,
                'alert_type': self.TYPE_SURGE,
                'severity': self.SEVERITY_CRITICAL if growth_rate > 100 else self.SEVERITY_HIGH,
                'title': f"🚨 Case Surge Detected: {region_name}",
                'description': f"Cases increased by {growth_rate:.1f}% in the last 3 days. Immediate attention required.",
                'metrics': {
                    'growth_rate': growth_rate,
                    'current_cases': growth_metrics['current_cases'],
                    'trend': trend
                },
                'triggered_at': datetime.now().isoformat(),
                'recommendations': [
                    "Implement enhanced public health measures",
                    "Increase testing and contact tracing capacity",
                    "Prepare hospitals for surge in admissions"
                ]
            })

        # Rapid growth alert (less severe than surge)
        elif growth_rate >= self.thresholds['rapid_growth_rate']:
            alerts.append({
                'region_id': region_id,
                'region_name': region_name,
                'alert_type': self.TYPE_RAPID_GROWTH,
                'severity': self.SEVERITY_MEDIUM,
                'title': f"⚠️ Rapid Growth: {region_name}",
                'description': f"Cases increasing at {growth_rate:.1f}% over 3 days. Monitor closely.",
                'metrics': {
                    'growth_rate': growth_rate,
                    'current_cases': growth_metrics['current_cases']
                },
                'triggered_at': datetime.now().isoformat(),
                'recommendations': [
                    "Increase surveillance in affected areas",
                    "Review hospital preparedness plans"
                ]
            })

        # Doubling time alert
        if doubling_time and doubling_time <= self.thresholds['doubling_time_critical']:
            alerts.append({
                'region_id': region_id,
                'region_name': region_name,
                'alert_type': self.TYPE_RAPID_GROWTH,
                'severity': self.SEVERITY_CRITICAL,
                'title': f"🔴 Critical Doubling Time: {region_name}",
                'description': f"Cases doubling every {doubling_time:.1f} days. Exponential growth detected.",
                'metrics': {
                    'doubling_time': doubling_time,
                    'daily_velocity': growth_metrics['daily_velocity']
                },
                'triggered_at': datetime.now().isoformat(),
                'recommendations': [
                    "Consider lockdown or movement restrictions",
                    "Activate emergency response protocols",
                    "Coordinate with neighboring regions"
                ]
            })

        return alerts

    def generate_capacity_alerts(self, region_name: str, region_id: str,
                                 resource_forecast: Dict, current_capacity: Dict) -> List[Dict]:
        """Generate alerts for hospital capacity issues.

        Args:
            region_name: Name of region
            region_id: Region identifier
            resource_forecast: Output from ResourceDemandPredictor
            current_capacity: Current hospital capacity

        Returns:
            List of alert dicts
        """
        alerts = []

        if not resource_forecast.get('success'):
            return alerts

        summary = resource_forecast.get('summary', {})
        max_utilization = summary.get('max_icu_utilization', 0)
        peak_date = summary.get('peak_date')
        peak_icu_needed = summary.get('peak_icu_beds_needed', 0)

        current_icu = current_capacity.get('icu_beds', 0)

        # Critical capacity alert
        if max_utilization >= self.thresholds['icu_utilization_critical']:
            gap = peak_icu_needed - current_icu

            alerts.append({
                'region_id': region_id,
                'region_name': region_name,
                'alert_type': self.TYPE_CAPACITY,
                'severity': self.SEVERITY_CRITICAL,
                'title': f"🏥 ICU Capacity Crisis: {region_name}",
                'description': f"ICU utilization will reach {max_utilization:.1f}% by {peak_date}. Shortage of {gap} beds predicted.",
                'metrics': {
                    'max_utilization': max_utilization,
                    'current_icu_beds': current_icu,
                    'peak_icu_needed': peak_icu_needed,
                    'gap': gap,
                    'peak_date': peak_date
                },
                'triggered_at': datetime.now().isoformat(),
                'recommendations': [
                    f"URGENT: Procure {gap} additional ICU beds before {peak_date}",
                    "Convert general wards to ICU capacity",
                    "Coordinate patient transfers to neighboring facilities",
                    "Activate field hospital protocols if available"
                ]
            })

        # Warning capacity alert
        elif max_utilization >= self.thresholds['icu_utilization_warning']:
            alerts.append({
                'region_id': region_id,
                'region_name': region_name,
                'alert_type': self.TYPE_CAPACITY,
                'severity': self.SEVERITY_HIGH,
                'title': f"⚠️ ICU Capacity Warning: {region_name}",
                'description': f"ICU utilization approaching {max_utilization:.1f}% by {peak_date}. Prepare surge capacity.",
                'metrics': {
                    'max_utilization': max_utilization,
                    'peak_date': peak_date
                },
                'triggered_at': datetime.now().isoformat(),
                'recommendations': [
                    "Activate surge capacity plans",
                    "Defer non-urgent procedures to free capacity",
                    "Review staffing levels and prepare for overtime"
                ]
            })

        # Ventilator shortage alert
        peak_ventilators = summary.get('peak_ventilators_needed', 0)
        current_ventilators = current_capacity.get('ventilators_available', 0)

        if peak_ventilators > current_ventilators:
            ventilator_gap = peak_ventilators - current_ventilators

            alerts.append({
                'region_id': region_id,
                'region_name': region_name,
                'alert_type': self.TYPE_RESOURCE_DEPLETION,
                'severity': self.SEVERITY_HIGH,
                'title': f"💨 Ventilator Shortage Predicted: {region_name}",
                'description': f"Need {peak_ventilators} ventilators by {peak_date}, but only {current_ventilators} available. Gap: {ventilator_gap}",
                'metrics': {
                    'peak_ventilators_needed': peak_ventilators,
                    'available': current_ventilators,
                    'gap': ventilator_gap,
                    'peak_date': peak_date
                },
                'triggered_at': datetime.now().isoformat(),
                'recommendations': [
                    f"Procure {ventilator_gap} additional ventilators immediately",
                    "Coordinate ventilator sharing with nearby hospitals",
                    "Train staff on ventilator operation and maintenance"
                ]
            })

        return alerts

    def generate_resource_depletion_alerts(self, region_name: str, region_id: str,
                                          resource_status: Dict) -> List[Dict]:
        """Generate alerts for depleting resources (oxygen, supplies).

        Args:
            region_name: Name of region
            region_id: Region identifier
            resource_status: Current resource levels

        Returns:
            List of alert dicts
        """
        alerts = []

        # Oxygen depletion alert
        oxygen_days = resource_status.get('oxygen_supply_days')

        if oxygen_days is not None:
            if oxygen_days <= self.thresholds['oxygen_days_critical']:
                alerts.append({
                    'region_id': region_id,
                    'region_name': region_name,
                    'alert_type': self.TYPE_RESOURCE_DEPLETION,
                    'severity': self.SEVERITY_CRITICAL,
                    'title': f"🔴 Oxygen Crisis: {region_name}",
                    'description': f"Oxygen supply will deplete in {oxygen_days} days at current consumption rate.",
                    'metrics': {
                        'oxygen_days_remaining': oxygen_days
                    },
                    'triggered_at': datetime.now().isoformat(),
                    'recommendations': [
                        "Emergency oxygen procurement required",
                        "Activate oxygen rationing protocols",
                        "Coordinate emergency oxygen transfers from other regions"
                    ]
                })

            elif oxygen_days <= self.thresholds['oxygen_days_warning']:
                alerts.append({
                    'region_id': region_id,
                    'region_name': region_name,
                    'alert_type': self.TYPE_RESOURCE_DEPLETION,
                    'severity': self.SEVERITY_MEDIUM,
                    'title': f"⚠️ Oxygen Supply Warning: {region_name}",
                    'description': f"Oxygen supply at {oxygen_days} days remaining. Replenishment needed.",
                    'metrics': {
                        'oxygen_days_remaining': oxygen_days
                    },
                    'triggered_at': datetime.now().isoformat(),
                    'recommendations': [
                        "Schedule oxygen delivery within 48 hours",
                        "Monitor oxygen consumption closely"
                    ]
                })

        return alerts

    def generate_all_alerts(self, region_name: str, region_id: str,
                           timeseries_data: List[Dict],
                           resource_forecast: Dict,
                           current_capacity: Dict,
                           resource_status: Dict) -> List[Dict]:
        """Generate all applicable alerts for a region.

        Args:
            region_name: Name of region
            region_id: Region identifier
            timeseries_data: Historical case data
            resource_forecast: Resource demand predictions
            current_capacity: Hospital capacity
            resource_status: Current resource levels

        Returns:
            List of all generated alerts, sorted by severity
        """
        all_alerts = []

        # Growth alerts
        all_alerts.extend(self.generate_growth_alerts(
            region_name, region_id, timeseries_data
        ))

        # Capacity alerts
        all_alerts.extend(self.generate_capacity_alerts(
            region_name, region_id, resource_forecast, current_capacity
        ))

        # Resource depletion alerts
        all_alerts.extend(self.generate_resource_depletion_alerts(
            region_name, region_id, resource_status
        ))

        # Sort by severity (critical first)
        severity_order = {
            self.SEVERITY_CRITICAL: 0,
            self.SEVERITY_HIGH: 1,
            self.SEVERITY_MEDIUM: 2,
            self.SEVERITY_LOW: 3
        }

        all_alerts.sort(key=lambda x: severity_order.get(x['severity'], 99))

        return all_alerts

    def get_alert_summary(self, alerts: List[Dict]) -> Dict:
        """Generate summary statistics for alerts.

        Args:
            alerts: List of alert dicts

        Returns:
            Dict with alert counts by severity and type
        """
        summary = {
            'total_alerts': len(alerts),
            'by_severity': {
                self.SEVERITY_CRITICAL: 0,
                self.SEVERITY_HIGH: 0,
                self.SEVERITY_MEDIUM: 0,
                self.SEVERITY_LOW: 0
            },
            'by_type': {},
            'most_urgent': alerts[0] if alerts else None
        }

        for alert in alerts:
            # Count by severity
            severity = alert.get('severity', self.SEVERITY_LOW)
            summary['by_severity'][severity] += 1

            # Count by type
            alert_type = alert.get('alert_type', 'unknown')
            summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1

        return summary
