"""
Machine Learning Models Module for Urban Heat Island Analysis

This module contains various ML models and evaluation functions
for predicting urban heat island intensity.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class HeatIslandPredictor:
    """
    A comprehensive machine learning pipeline for urban heat island prediction.
    """

    def __init__(self, models=None, scaling_method='standard'):
        """
        Initialize the predictor with multiple models.

        Parameters:
        -----------
        models : dict
            Dictionary of model name: model object pairs
        scaling_method : str
            'standard', 'minmax', or 'none'
        """
        self.models = models or self._get_default_models()
        self.scaling_method = scaling_method
        self.scaler = None
        self.results = {}
        self.best_model = None
        self.feature_names = None

    def _get_default_models(self):
        """Get default set of models for comparison."""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=5,
                learning_rate=0.1
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=500,
                alpha=0.01
            )
        }

    def prepare_data(self, data, feature_columns, target_column, test_size=0.2):
        """
        Prepare data for machine learning.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        feature_columns : list
            List of feature column names
        target_column : str
            Target variable column name
        test_size : float
            Proportion of data for testing

        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test
        """
        self.feature_names = feature_columns

        # Select features and target
        X = data[feature_columns].copy()
        y = data[target_column].copy()

        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features if requested
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, y_train, cv_folds=5):
        """
        Train all models and perform cross-validation.

        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        cv_folds : int
            Number of cross-validation folds
        """
        print("🤖 Training models...")

        for name, model in self.models.items():
            print(f"   Training {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv_folds, scoring='r2'
            )

            # Store results
            self.results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }

            print(f"     CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models on test set.

        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        """
        print("\n📊 Evaluating models on test set...")

        for name in self.results:
            model = self.results[name]['model']

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store test results
            self.results[name].update({
                'y_pred': y_pred,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })

            print(f"   {name:20} | R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

        # Find best model
        best_r2 = max(self.results[name]['r2'] for name in self.results)
        self.best_model = [name for name in self.results 
                          if self.results[name]['r2'] == best_r2][0]

        print(f"\n🏆 Best Model: {self.best_model} (R² = {best_r2:.4f})")

    def get_feature_importance(self, top_n=None):
        """
        Get feature importance from tree-based models.

        Parameters:
        -----------
        top_n : int
            Number of top features to return

        Returns:
        --------
        dict
            Feature importance for each applicable model
        """
        importance_dict = {}

        for name, result in self.results.items():
            model = result['model']

            # Check if model has feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)

                if top_n:
                    feature_imp = feature_imp.head(top_n)

                importance_dict[name] = feature_imp

        return importance_dict

    def hyperparameter_tuning(self, X_train, y_train, model_name='Random Forest'):
        """
        Perform hyperparameter tuning for a specific model.

        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        model_name : str
            Name of model to tune
        """
        print(f"🔧 Tuning hyperparameters for {model_name}...")

        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }

        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return

        # Get base model
        base_model = self.models[model_name]

        # Perform grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grids[model_name],
            cv=3, 
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Update model with best parameters
        self.models[model_name + '_Tuned'] = grid_search.best_estimator_

        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def predict_scenarios(self, scenarios_df):
        """
        Make predictions for different scenarios.

        Parameters:
        -----------
        scenarios_df : pandas.DataFrame
            DataFrame with feature values for different scenarios

        Returns:
        --------
        pandas.DataFrame
            Scenarios with predictions from all models
        """
        results_df = scenarios_df.copy()

        # Prepare features
        X_scenarios = scenarios_df[self.feature_names].values

        # Scale if needed
        if self.scaler:
            X_scenarios = self.scaler.transform(X_scenarios)

        # Make predictions with all models
        for name, result in self.results.items():
            model = result['model']
            predictions = model.predict(X_scenarios)
            results_df[f'{name}_prediction'] = predictions

        return results_df

    def save_results(self, filepath='model_results.csv'):
        """Save model comparison results to CSV."""
        results_list = []

        for name, result in self.results.items():
            results_list.append({
                'model': name,
                'cv_r2_mean': result.get('cv_mean', np.nan),
                'cv_r2_std': result.get('cv_std', np.nan),
                'test_r2': result.get('r2', np.nan),
                'test_rmse': result.get('rmse', np.nan),
                'test_mae': result.get('mae', np.nan)
            })

        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('test_r2', ascending=False)
        results_df.to_csv(filepath, index=False)
        print(f"✅ Model results saved to {filepath}")

def create_ensemble_model(models_dict, weights=None):
    """
    Create an ensemble model from multiple trained models.

    Parameters:
    -----------
    models_dict : dict
        Dictionary of trained models
    weights : list
        Optional weights for ensemble averaging

    Returns:
    --------
    callable
        Ensemble prediction function
    """
    if weights is None:
        weights = [1.0] * len(models_dict)

    def ensemble_predict(X):
        predictions = []
        for i, (name, model) in enumerate(models_dict.items()):
            pred = model.predict(X) * weights[i]
            predictions.append(pred)

        return np.mean(predictions, axis=0)

    return ensemble_predict

if __name__ == "__main__":
    # Example usage
    print("🤖 Testing ML models module...")

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    # Features
    X = pd.DataFrame({
        'ndvi': np.random.uniform(-0.1, 0.8, n_samples),
        'building_density': np.random.uniform(0, 1, n_samples),
        'population_density': np.random.lognormal(8, 1, n_samples),
        'elevation': np.random.normal(20, 10, n_samples)
    })

    # Target (temperature) with realistic relationships
    y = (25 - 8 * X['ndvi'] + 6 * X['building_density'] + 
         0.0001 * X['population_density'] - 0.1 * X['elevation'] +
         np.random.normal(0, 1.5, n_samples))

    data = X.copy()
    data['temperature'] = y

    # Initialize predictor
    predictor = HeatIslandPredictor()

    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(
        data, 
        feature_columns=['ndvi', 'building_density', 'population_density', 'elevation'],
        target_column='temperature'
    )

    # Train models
    predictor.train_models(X_train, y_train)

    # Evaluate models
    predictor.evaluate_models(X_test, y_test)

    # Get feature importance
    importance = predictor.get_feature_importance()
    if importance:
        print("\n🌟 Feature Importance (Random Forest):")
        for _, row in importance['Random Forest'].iterrows():
            print(f"   {row['feature']:20} | {row['importance']:.4f}")

    print("\n✅ ML models module test completed!")
