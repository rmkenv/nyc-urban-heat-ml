"""
Data Processing Module for NYC Urban Heat Island Analysis

This module contains functions for loading, cleaning, and processing
satellite imagery and urban data for heat island analysis.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_sample_landsat_data(bounds, n_points=1000, seed=42):
    """
    Create sample Landsat-like data for demonstration purposes.

    In a real project, this would be replaced with actual satellite data
    download functions using APIs like Google Earth Engine or USGS.

    Parameters:
    -----------
    bounds : dict
        Dictionary with lat_min, lat_max, lon_min, lon_max
    n_points : int
        Number of sample points to generate
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pandas.DataFrame
        Sample satellite data with bands and indices
    """
    np.random.seed(seed)

    # Generate coordinates
    lats = np.random.uniform(bounds['lat_min'], bounds['lat_max'], n_points)
    lons = np.random.uniform(bounds['lon_min'], bounds['lon_max'], n_points)

    # Simulate Landsat bands (scaled 0-1 for simplicity)
    red = np.random.beta(2, 3, n_points)  # Red band
    nir = np.random.beta(3, 2, n_points)  # Near Infrared band
    green = np.random.beta(2.5, 2.5, n_points)  # Green band
    blue = np.random.beta(2, 3, n_points)  # Blue band
    swir = np.random.beta(2, 4, n_points)  # Short Wave Infrared
    thermal = np.random.beta(3, 2, n_points)  # Thermal band

    # Calculate indices
    ndvi = (nir - red) / (nir + red + 1e-8)  # Vegetation index
    ndbi = (swir - nir) / (swir + nir + 1e-8)  # Built-up index
    ndwi = (green - nir) / (green + nir + 1e-8)  # Water index

    # Land Surface Temperature (derived from thermal band)
    lst_kelvin = 273.15 + thermal * 50 + 273  # Simulate LST in Kelvin
    lst_celsius = lst_kelvin - 273.15

    return pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'red': red,
        'nir': nir,
        'green': green,
        'blue': blue,
        'swir': swir,
        'thermal': thermal,
        'ndvi': ndvi,
        'ndbi': ndbi,
        'ndwi': ndwi,
        'lst_kelvin': lst_kelvin,
        'lst_celsius': lst_celsius
    })

def download_nyc_boundaries():
    """
    Download NYC borough boundaries from NYC Open Data.

    Returns:
    --------
    geopandas.GeoDataFrame
        NYC borough boundaries
    """
    try:
        # NYC Borough Boundaries
        url = "https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=GeoJSON"
        gdf = gpd.read_file(url)
        return gdf
    except Exception as e:
        print(f"Could not download NYC boundaries: {e}")
        # Return a simple placeholder
        return create_sample_boundaries()

def create_sample_boundaries():
    """Create sample NYC-like boundaries for demonstration."""
    from shapely.geometry import Polygon

    # Approximate NYC borough boundaries
    boroughs = {
        'Manhattan': Polygon([(-74.02, 40.68), (-73.97, 40.68), (-73.93, 40.80), (-74.02, 40.80)]),
        'Brooklyn': Polygon([(-74.05, 40.57), (-73.86, 40.57), (-73.86, 40.73), (-74.05, 40.73)]),
        'Queens': Polygon([(-73.96, 40.54), (-73.70, 40.54), (-73.70, 40.80), (-73.96, 40.80)]),
        'Bronx': Polygon([(-73.93, 40.78), (-73.75, 40.78), (-73.75, 40.92), (-73.93, 40.92)]),
        'Staten Island': Polygon([(-74.26, 40.50), (-74.05, 40.50), (-74.05, 40.65), (-74.26, 40.65)])
    }

    return gpd.GeoDataFrame(
        {'borough': list(boroughs.keys()), 'geometry': list(boroughs.values())},
        crs='EPSG:4326'
    )

def calculate_urban_features(data):
    """
    Calculate urban heat island related features from satellite data.

    Parameters:
    -----------
    data : pandas.DataFrame
        Satellite data with spectral bands

    Returns:
    --------
    pandas.DataFrame
        Data with additional urban features
    """
    df = data.copy()

    # Urban Heat Island Intensity (simplified)
    baseline_temp = df['lst_celsius'].quantile(0.1)  # Rural reference
    df['uhi_intensity'] = df['lst_celsius'] - baseline_temp

    # Vegetation health categories
    df['vegetation_category'] = pd.cut(
        df['ndvi'], 
        bins=[-1, 0.1, 0.3, 0.5, 1.0],
        labels=['Sparse', 'Low', 'Moderate', 'Dense']
    )

    # Built-up intensity categories  
    df['buildup_category'] = pd.cut(
        df['ndbi'],
        bins=[-1, -0.1, 0.1, 0.3, 1.0],
        labels=['Natural', 'Low', 'Medium', 'High']
    )

    # Temperature categories
    df['temp_category'] = pd.cut(
        df['lst_celsius'],
        bins=[df['lst_celsius'].min()-1, 
              df['lst_celsius'].quantile(0.25),
              df['lst_celsius'].quantile(0.75),
              df['lst_celsius'].max()+1],
        labels=['Cool', 'Moderate', 'Hot']
    )

    return df

def create_spatial_features(data, grid_size=0.01):
    """
    Create spatial aggregation features.

    Parameters:
    -----------
    data : pandas.DataFrame
        Point data with lat/lon
    grid_size : float
        Size of spatial grid in degrees

    Returns:
    --------
    pandas.DataFrame
        Data with spatial features
    """
    df = data.copy()

    # Create spatial grid
    df['grid_lat'] = (df['latitude'] // grid_size) * grid_size
    df['grid_lon'] = (df['longitude'] // grid_size) * grid_size

    # Calculate neighborhood statistics
    grid_stats = df.groupby(['grid_lat', 'grid_lon']).agg({
        'lst_celsius': ['mean', 'std', 'count'],
        'ndvi': ['mean', 'std'],
        'ndbi': ['mean', 'std']
    }).round(3)

    grid_stats.columns = ['_'.join(col).strip() for col in grid_stats.columns.values]
    grid_stats = grid_stats.reset_index()

    # Merge back to original data
    df = df.merge(grid_stats, on=['grid_lat', 'grid_lon'], how='left')

    return df

def export_processed_data(data, output_path='processed_data.csv'):
    """
    Export processed data to CSV file.

    Parameters:
    -----------
    data : pandas.DataFrame
        Processed data
    output_path : str
        Output file path
    """
    # Select key columns for export
    export_cols = [
        'latitude', 'longitude', 'lst_celsius', 'ndvi', 'ndbi', 'ndwi',
        'uhi_intensity', 'vegetation_category', 'buildup_category', 'temp_category'
    ]

    # Add spatial features if they exist
    spatial_cols = [col for col in data.columns if 'mean' in col or 'std' in col]
    export_cols.extend(spatial_cols)

    # Filter to existing columns
    export_cols = [col for col in export_cols if col in data.columns]

    data[export_cols].to_csv(output_path, index=False)
    print(f"✅ Exported processed data to {output_path}")
    print(f"📊 Shape: {data.shape}")
    print(f"📝 Columns: {len(export_cols)}")

if __name__ == "__main__":
    # Example usage
    print("🔄 Running data processing example...")

    # NYC bounds
    nyc_bounds = {
        'lat_min': 40.4774, 'lat_max': 40.9176,
        'lon_min': -74.2591, 'lon_max': -73.7004
    }

    # Create sample data
    data = create_sample_landsat_data(nyc_bounds, n_points=500)
    print(f"📊 Created sample data: {data.shape}")

    # Add urban features
    data = calculate_urban_features(data)
    print(f"🏙️ Added urban features")

    # Add spatial features
    data = create_spatial_features(data)
    print(f"🗺️ Added spatial features")

    # Export data
    export_processed_data(data, 'sample_processed_data.csv')

    print("✅ Data processing example completed!")
