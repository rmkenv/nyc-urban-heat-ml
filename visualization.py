"""
Visualization Module for Urban Heat Island Analysis

This module contains functions for creating maps, charts, and interactive
visualizations for urban heat island data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

class HeatIslandVisualizer:
    """
    A comprehensive visualization toolkit for urban heat island analysis.
    """

    def __init__(self, figsize=(12, 8), dpi=100):
        """
        Initialize the visualizer.

        Parameters:
        -----------
        figsize : tuple
            Default figure size
        dpi : int
            Figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        self.color_maps = {
            'temperature': 'RdYlBu_r',
            'vegetation': 'RdYlGn',
            'buildup': 'gray',
            'water': 'Blues'
        }

    def plot_correlation_matrix(self, data, columns=None, title="Correlation Matrix"):
        """
        Create a correlation heatmap.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        columns : list
            Columns to include in correlation matrix
        title : str
            Plot title
        """
        if columns:
            corr_data = data[columns]
        else:
            corr_data = data.select_dtypes(include=[np.number])

        correlation = corr_data.corr()

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation, dtype=bool))

        # Generate heatmap
        sns.heatmap(correlation, mask=mask, annot=True, cmap='RdBu_r',
                   center=0, square=True, fmt='.3f', cbar_kws={"shrink": .8})

        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        return fig

    def plot_temperature_distribution(self, data, temp_column='temperature'):
        """
        Create temperature distribution plots.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        temp_column : str
            Temperature column name
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)

        # Histogram
        axes[0, 0].hist(data[temp_column], bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].set_title('Temperature Distribution')
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('Frequency')

        # Box plot
        sns.boxplot(y=data[temp_column], ax=axes[0, 1], color='lightcoral')
        axes[0, 1].set_title('Temperature Box Plot')
        axes[0, 1].set_ylabel('Temperature (°C)')

        # QQ plot (approximate)
        from scipy import stats
        stats.probplot(data[temp_column], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')

        # Density plot
        data[temp_column].plot.density(ax=axes[1, 1], color='red', alpha=0.7)
        axes[1, 1].set_title('Temperature Density')
        axes[1, 1].set_xlabel('Temperature (°C)')

        plt.suptitle('🌡️ Temperature Analysis', fontsize=16)
        plt.tight_layout()
        return fig

    def plot_spatial_patterns(self, data, lat_col='latitude', lon_col='longitude', 
                            value_col='temperature', title='Spatial Temperature Pattern'):
        """
        Create spatial scatter plot.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        lat_col, lon_col : str
            Latitude and longitude column names
        value_col : str
            Value column for color coding
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        scatter = ax.scatter(data[lon_col], data[lat_col], 
                           c=data[value_col], cmap=self.color_maps['temperature'],
                           s=20, alpha=0.7)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        if 'temp' in value_col.lower():
            cbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
        else:
            cbar.set_label(value_col, rotation=270, labelpad=20)

        plt.tight_layout()
        return fig

    def plot_ndvi_temperature_relationship(self, data, ndvi_col='ndvi', temp_col='temperature'):
        """
        Plot NDVI vs temperature relationship.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        ndvi_col : str
            NDVI column name
        temp_col : str
            Temperature column name
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Scatter plot
        ax.scatter(data[ndvi_col], data[temp_col], alpha=0.6, s=20)

        # Add trend line
        z = np.polyfit(data[ndvi_col], data[temp_col], 1)
        p = np.poly1d(z)
        ax.plot(data[ndvi_col], p(data[ndvi_col]), "r--", alpha=0.8, linewidth=2)

        # Calculate correlation
        correlation = data[ndvi_col].corr(data[temp_col])

        ax.set_xlabel('NDVI (Vegetation Index)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('🌱 Vegetation vs Temperature Relationship')

        # Add correlation text
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def create_interactive_heat_map(self, data, lat_col='latitude', lon_col='longitude',
                                  temp_col='temperature', center_lat=40.7128, center_lon=-74.0060):
        """
        Create interactive folium heat map.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        lat_col, lon_col : str
            Latitude and longitude column names
        temp_col : str
            Temperature column name
        center_lat, center_lon : float
            Map center coordinates

        Returns:
        --------
        folium.Map
            Interactive map object
        """
        # Create base map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        # Prepare heat data
        heat_data = [[row[lat_col], row[lon_col], row[temp_col]] 
                    for idx, row in data.iterrows()]

        # Add heat map layer
        heat_map = plugins.HeatMap(heat_data, radius=15, blur=10, gradient={
            0.0: 'blue', 0.3: 'cyan', 0.5: 'lime', 0.7: 'yellow', 1.0: 'red'
        })
        heat_map.add_to(m)

        # Add sample points with popups
        temp_percentiles = data[temp_col].quantile([0.2, 0.4, 0.6, 0.8])

        for idx, row in data.sample(min(50, len(data))).iterrows():
            # Color code based on temperature
            if row[temp_col] > temp_percentiles[0.8]:
                color = 'red'
                icon = '🔥'
            elif row[temp_col] > temp_percentiles[0.6]:
                color = 'orange'
                icon = '🌡️'
            elif row[temp_col] > temp_percentiles[0.4]:
                color = 'yellow'
                icon = '☀️'
            else:
                color = 'blue'
                icon = '❄️'

            # Create popup
            popup_text = f"""
            <b>{icon} Temperature Point</b><br>
            🌡️ Temperature: {row[temp_col]:.1f}°C<br>
            📍 Location: ({row[lat_col]:.4f}, {row[lon_col]:.4f})
            """

            if 'ndvi' in data.columns:
                popup_text += f"<br>🌱 NDVI: {row['ndvi']:.3f}"

            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=6,
                popup=popup_text,
                color=color,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)

        return m

    def plot_model_comparison(self, results_dict):
        """
        Plot model performance comparison.

        Parameters:
        -----------
        results_dict : dict
            Dictionary with model results
        """
        models = list(results_dict.keys())
        r2_scores = [results_dict[model].get('r2', 0) for model in models]
        rmse_scores = [results_dict[model].get('rmse', 0) for model in models]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)

        # R² scores
        bars1 = axes[0].bar(models, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[0].set_title('📊 Model Performance: R² Score')
        axes[0].set_ylabel('R² Score')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)

        for bar, score in zip(bars1, r2_scores):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')

        # RMSE scores
        bars2 = axes[1].bar(models, rmse_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[1].set_title('📉 Model Performance: RMSE')
        axes[1].set_ylabel('RMSE (°C)')
        axes[1].tick_params(axis='x', rotation=45)

        for bar, score in zip(bars2, rmse_scores):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        return fig


def save_plot(fig, filename, dpi=300, bbox_inches='tight'):
    """Save plot with high quality."""
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"✅ Saved plot: {filename}")

if __name__ == "__main__":
    # Example usage
    print("📊 Testing visualization module...")

    # Create sample data
    np.random.seed(42)
    n_samples = 500

    data = pd.DataFrame({
        'latitude': np.random.uniform(40.5, 40.9, n_samples),
        'longitude': np.random.uniform(-74.2, -73.7, n_samples),
        'temperature': np.random.normal(25, 3, n_samples),
        'ndvi': np.random.uniform(-0.1, 0.8, n_samples)
    })

    # Create visualizer
    viz = HeatIslandVisualizer()

    # Test plots
    print("   Creating correlation matrix...")
    corr_fig = viz.plot_correlation_matrix(data)

    print("   Creating temperature distribution...")
    temp_fig = viz.plot_temperature_distribution(data)

    print("   Creating NDVI-temperature relationship...")
    ndvi_fig = viz.plot_ndvi_temperature_relationship(data)

    print("✅ Visualization module test completed!")
