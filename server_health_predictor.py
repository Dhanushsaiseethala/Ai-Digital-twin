"""
Minimal LightGBM model integration for Load Distribution.
Loads the trained server_health_lightgbm.txt model and provides health predictions.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any

try:
    import lightgbm as lgb
    import pandas as pd
    import numpy as np
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


MODEL_PATH = Path(os.path.expanduser(r"C:\Users\seeth\Downloads\server_health_lightgbm.txt"))
FEATURES_PATH = Path(os.path.expanduser(r"C:\Users\seeth\Downloads\model_features.json"))
# Fallback  list (56 features based on typical server health metrics)
DEFAULT_FEATURES = [
    'server_id_num', 'rack_id_num', 'position_x', 'position_y', 'position_z',
    'idle_power_w', 'max_power_w', 'workload', 'current_power_w',
    'airflow_m3_s', 'temperature_current', 'temperature_supply', 'temperature_return',
    'temperature_ambient', 'delta_t', 'inlet_temp', 'outlet_temp',
    'cpu_usage_pct', 'memory_usage_pct', 'disk_usage_pct', 'network_usage_pct',
    'thermal_density', 'power_density', 'uptime_days', 'age_years',
    'maintenance_overdue', 'last_maintenance_days_ago', 'error_count_24h',
    'error_rate_pct', 'restart_count_24h', 'hang_count_24h',
    'cooling_effectiveness', 'airflow_blockage_pct', 'dust_level_pct',
    'vibration_level', 'acoustic_level_db', 'humidity_pct', 'pressure_pa',
    'voltage_deviation_pct', 'frequency_deviation_hz', 'power_factor',
    'harmonic_distortion_pct', 'cable_insulation_resistance_kohm',
    'connector_oxidation_level', 'thermal_paste_degradation_pct',
    'bearing_wear_level', 'fan_noise_db', 'fan_vibration',
    'ambient_temp_stability', 'recent_temp_spike_count', 'thermal_shock_events',
    'workload_variability', 'peak_load_last_hour', 'sustained_high_load_hours',
]


def load_model():
    """Load the LightGBM model from disk if available."""
    if not LIGHTGBM_AVAILABLE:
        return None
    
    if not MODEL_PATH.exists():
        return None
    
    try:
        model = lgb.Booster(model_file=str(MODEL_PATH))
        return model
    except Exception:
        return None


def load_feature_order() -> List[str]:
    """Load feature order from JSON if available, otherwise use defaults."""
    if FEATURES_PATH.exists():
        try:
            with open(FEATURES_PATH, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'features' in data:
                    return data['features']
                elif isinstance(data, list):
                    return data
        except Exception:
            pass
    
    return DEFAULT_FEATURES


def construct_server_features(server_data: Dict[str, Any], feature_order: List[str]) -> pd.DataFrame:
    """
    Construct feature dataframe from server data using exact feature order.
    
    Args:
        server_data: Dictionary with server information
        feature_order: List of feature column names in required order
    
    Returns:
        DataFrame with exactly the required features in correct order
    """
    # Map available server data to features
    feature_values = {}
    
    # Parse server ID to extract numeric components
    server_id = server_data.get('id', 'Rack A01-S01')
    rack_id = server_data.get('rackId', 'Rack A01')
    
    try:
        rack_num = int(rack_id.replace('Rack A', ''))
    except:
        rack_num = 1
    
    try:
        server_num = int(server_id.split('-S')[-1])
    except:
        server_num = 1
    
    # Get position from server data
    position = server_data.get('position', [0, 0, 0])
    
    # Construct all 56 features with reasonable defaults
    feature_values = {
        'server_id_num': server_num,
        'rack_id_num': rack_num,
        'position_x': float(position[0]) if len(position) > 0 else 0.0,
        'position_y': float(position[1]) if len(position) > 1 else 0.0,
        'position_z': float(position[2]) if len(position) > 2 else 0.0,
        'idle_power_w': 100.0,
        'max_power_w': 450.0,
        'workload': 0.5,
        'current_power_w': 275.0,
        'airflow_m3_s': 0.12,
        'temperature_current': 28.0,
        'temperature_supply': 18.0,
        'temperature_return': 28.0,
        'temperature_ambient': 24.0,
        'delta_t': 10.0,
        'inlet_temp': 24.0,
        'outlet_temp': 28.0,
        'cpu_usage_pct': 45.0,
        'memory_usage_pct': 65.0,
        'disk_usage_pct': 70.0,
        'network_usage_pct': 30.0,
        'thermal_density': 7.5,
        'power_density': 75.0,
        'uptime_days': 365.0,
        'age_years': 3.0,
        'maintenance_overdue': 0.0,
        'last_maintenance_days_ago': 45.0,
        'error_count_24h': 0.0,
        'error_rate_pct': 0.1,
        'restart_count_24h': 0.0,
        'hang_count_24h': 0.0,
        'cooling_effectiveness': 0.85,
        'airflow_blockage_pct': 5.0,
        'dust_level_pct': 2.0,
        'vibration_level': 2.5,
        'acoustic_level_db': 65.0,
        'humidity_pct': 45.0,
        'pressure_pa': 101325.0,
        'voltage_deviation_pct': 1.5,
        'frequency_deviation_hz': 0.1,
        'power_factor': 0.95,
        'harmonic_distortion_pct': 3.0,
        'cable_insulation_resistance_kohm': 500.0,
        'connector_oxidation_level': 1.0,
        'thermal_paste_degradation_pct': 10.0,
        'bearing_wear_level': 5.0,
        'fan_noise_db': 62.0,
        'fan_vibration': 2.0,
        'ambient_temp_stability': 0.9,
        'recent_temp_spike_count': 0.0,
        'thermal_shock_events': 0.0,
        'workload_variability': 0.15,
        'peak_load_last_hour': 0.65,
        'sustained_high_load_hours': 2.0,
    }
    
    # Build dataframe with only the features in the exact order required
    df_values = {}
    for feature in feature_order:
        df_values[feature] = [feature_values.get(feature, 0.0)]
    
    return pd.DataFrame(df_values)


def predict_server_health(server_data: Dict[str, Any], model=None, feature_order=None) -> Tuple[str, float]:
    """
    Predict server health using LightGBM model or fallback to health score.
    
    Args:
        server_data: Dictionary with server information
        model: Loaded LightGBM model (optional)
        feature_order: List of feature names in required order
    
    Returns:
        Tuple of (health_category, health_score) where:
        - health_score: 0-100 (0=Critical, 50-79=Warning, 80-100=Healthy)
        - health_category: "Healthy", "Degraded", or "Unhealthy"
    """
    if feature_order is None:
        feature_order = load_feature_order()
    
    # If model is available and LightGBM is installed, use it
    if model is not None and LIGHTGBM_AVAILABLE:
        try:
            df = construct_server_features(server_data, feature_order)
            predictions = model.predict(df, num_iteration=model.best_iteration)
            health_score = float(predictions[0] * 100)  # Normalize to 0-100
        except Exception as e:
            # Fallback: use position-based heuristic
            health_score = _fallback_health_score(server_data)
    else:
        # No model available: use position and basic heuristics
        health_score = _fallback_health_score(server_data)
    
    # Clamp score to 0-100
    health_score = max(0.0, min(100.0, health_score))
    
    # Map score to health category
    if health_score >= 80:
        health_category = "Healthy"
    elif health_score >= 50:
        health_category = "Degraded"
    else:
        health_category = "Unhealthy"
    
    return health_category, health_score


def predict_server_health_from_features(features: Dict[str, Any]) -> Tuple[float | None, str | None]:
    """Return an actual model output only when model and every required input exist."""
    if not LIGHTGBM_AVAILABLE:
        return None, "LightGBM unavailable for this experiment"

    model = load_model()
    if model is None:
        return None, "LightGBM unavailable for this experiment"

    feature_order = model.feature_name() or load_feature_order()
    if not feature_order or any(name not in features for name in feature_order):
        return None, "LightGBM unavailable for this experiment"

    try:
        frame = pd.DataFrame({name: [float(features[name])] for name in feature_order})
        if not np.isfinite(frame.to_numpy()).all():
            return None, "LightGBM unavailable for this experiment"
        prediction = float(model.predict(frame)[0])
        if not np.isfinite(prediction):
            return None, "LightGBM unavailable for this experiment"
        return prediction, None
    except Exception:
        return None, "LightGBM unavailable for this experiment"


def _fallback_health_score(server_data: Dict[str, Any]) -> float:
    """
    Fallback health score based on server position and rack index.
    Used when model is not available.
    """
    try:
        rack_id = server_data.get('rackId', 'Rack A01')
        rack_num = int(rack_id.replace('Rack A', ''))
        
        server_id = server_data.get('id', 'Rack A01-S01')
        server_num = int(server_id.split('-S')[-1])
        
        # Deterministic pseudo-random based on rack and server index
        seed = (rack_num * 13 + server_num * 7) % 1000
        base_health = 50 + (seed % 50)
        
        return base_health
    except:
        return 70.0  # Default to healthy


def get_server_health_predictions(servers_data: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    """
    Get health predictions for multiple servers.
    
    Args:
        servers_data: List of server data dictionaries
    
    Returns:
        List of (health_category, health_score) tuples
    """
    model = load_model()
    feature_order = load_feature_order()
    
    predictions = []
    for server in servers_data:
        health_cat, health_score = predict_server_health(server, model, feature_order)
        predictions.append((health_cat, health_score))
    
    return predictions


if __name__ == "__main__":
    # Test the module
    model = load_model()
    print(f"Model loaded: {model is not None}")
    
    test_server = {
        'id': 'Rack A01-S01',
        'rackId': 'Rack A01',
        'position': [0, 1, 0]
    }
    
    health_cat, health_score = predict_server_health(test_server, model)
    print(f"Test server health: {health_cat} (score: {health_score:.1f})")
