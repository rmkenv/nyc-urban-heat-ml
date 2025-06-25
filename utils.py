"""
Utility Functions for Urban Heat Island Analysis

This module contains helper functions for data processing,
file handling, and common calculations.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

def create_project_structure(base_path='.'):
    """Create the standard project directory structure."""
    directories = [
        'data/raw',
        'data/processed', 
        'data/sample_data',
        'notebooks',
        'src',
        'results/figures',
        'results/models',
        'docs'
    ]

    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def get_default_config():
    """Get default configuration settings."""
    return {
        'study_area': {
            'name': 'New York City',
            'bounds': {
                'lat_min': 40.4774,
                'lat_max': 40.9176,
                'lon_min': -74.2591,
                'lon_max': -73.7004
            }
        },
        'data_sources': {
            'satellite': 'Landsat_8_9',
            'weather': 'NOAA_stations',
            'urban': 'NYC_open_data'
        },
        'processing': {
            'cloud_threshold': 30,
            'spatial_resolution': 30,
            'temporal_window': 30
        },
        'ml_settings': {
            'test_size': 0.2,
            'cv_folds': 5,
            'random_state': 42
        }
    }

def calculate_uhi_intensity(temperature_data, reference_method='percentile', percentile=0.1):
    """Calculate Urban Heat Island intensity."""
    temp_array = np.array(temperature_data)

    if reference_method == 'percentile':
        reference_temp = np.percentile(temp_array, percentile * 100)
    elif reference_method == 'minimum':
        reference_temp = np.min(temp_array)
    else:
        reference_temp = np.percentile(temp_array, percentile * 100)

    uhi_intensity = temp_array - reference_temp
    return uhi_intensity

def classify_vegetation_health(ndvi_values):
    """Classify vegetation health based on NDVI values."""
    categories = []
    for ndvi in ndvi_values:
        if ndvi > 0.5:
            categories.append('Dense')
        elif ndvi > 0.3:
            categories.append('Moderate')
        elif ndvi > 0.1:
            categories.append('Sparse')
        else:
            categories.append('Very Sparse')

    return categories

def generate_report_summary(data, model_results=None):
    """Generate a summary report of the analysis."""
    summary = {
        'dataset': {
            'total_points': len(data),
            'study_area': 'New York City',
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        },
        'temperature': {
            'mean': data['temperature'].mean() if 'temperature' in data.columns else None,
            'min': data['temperature'].min() if 'temperature' in data.columns else None,
            'max': data['temperature'].max() if 'temperature' in data.columns else None,
            'std': data['temperature'].std() if 'temperature' in data.columns else None
        }
    }

    if 'ndvi' in data.columns:
        summary['vegetation'] = {
            'mean_ndvi': data['ndvi'].mean(),
            'ndvi_temp_correlation': data['ndvi'].corr(data['temperature']) if 'temperature' in data.columns else None
        }

    if model_results:
        best_model = max(model_results.keys(), key=lambda k: model_results[k].get('r2', 0))
        summary['best_model'] = {
            'name': best_model,
            'r2_score': model_results[best_model].get('r2', 0),
            'rmse': model_results[best_model].get('rmse', 0)
        }

    return summary

if __name__ == "__main__":
    print("🔧 Utility functions module ready!")
