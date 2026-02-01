# NFL Rushing Predictor Model
# Main ML model for predicting NFL rushing performance


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config_settings import MODEL_CONFIG, MODELS_DIR, PREDICTION_CONFIG
from src.data.data_processor import NFLDataProcessor
from .utils_helpers import update_predictions_with_weekly_stats

class NFLRushingPredictor:
    #Main predictor class for NFL rushing yards
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        self.model_type = model_type
        self.model = None
        self.processor = NFLDataProcessor()
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize model based on type
        if model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**MODEL_CONFIG)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=MODEL_CONFIG['n_estimators'],
                max_depth=MODEL_CONFIG['max_depth'],
                random_state=MODEL_CONFIG['random_state'],
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, df: pd.DataFrame, target_col: str = 'rushing_yards') -> Dict[str, float]:
        #Train the model on historical data
        self.logger.info(f"Training {self.model_type} model...")
        
        # Clean and validate data
        df = self.processor.clean_data(df)
        validation_results = self.processor.validate_data(df)
        self.logger.info(f"Data validation: {validation_results['total_records']} clean records")
        
        # Engineer features
        df = self.processor.engineer_features(df)
        
        # Prepare features and target
        X = self.processor.prepare_features(df, fit_encoders=True)
        y = self.processor.create_target_variable(df, target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'], 
            random_state=MODEL_CONFIG['random_state'],
            stratify=None
        )
        
        # Scale features
        X_train_scaled = self.processor.scale_features(X_train, fit_scaler=True)
        X_test_scaled = self.processor.scale_features(X_test, fit_scaler=False)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        metrics['cv_mae'] = -cv_scores.mean()
        metrics['cv_mae_std'] = cv_scores.std()
        
        self.logger.info(f"Model training completed. MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        #Make predictions for new data
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.logger.info(f"Making predictions for {len(df)} players...")
        
        # Engineer features (don't fit encoders for new data)
        df_processed = self.processor.engineer_features(df)
        X = self.processor.prepare_features(df_processed, fit_encoders=False)
        X_scaled = self.processor.scale_features(X, fit_scaler=False)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Create results dataframe
        results = df.copy()
        results['predicted_rushing_yards'] = predictions.astype(int)
        
        # Add confidence intervals if supported
        if hasattr(self.model, 'predict') and PREDICTION_CONFIG['confidence_intervals']:
            results = self._add_confidence_intervals(results, X_scaled)
        
        # Add injury risk assessment
        if PREDICTION_CONFIG['include_injury_risk']:
            results['injury_risk'] = self._assess_injury_risk(df_processed)
        
        # Sort by predicted yards
        results = results.sort_values('predicted_rushing_yards', ascending=False)
        
        self.logger.info("Predictions completed successfully")
        return results
    
    def predict_top_n(self, df: pd.DataFrame, n: int = None) -> pd.DataFrame:
        """Predict top N rushing leaders"""
        if n is None:
            n = PREDICTION_CONFIG['top_n_predictions']
        
        results = self.predict(df)
        return results.head(n)
    
    def get_feature_importance(self) -> pd.DataFrame:
        #Get feature importance from trained model
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(f"Model type {self.model_type} doesn't support feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.processor.get_feature_importance_names(),
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str = None) -> str:
        #Save trained model and processor
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            filepath = MODELS_DIR / f"nfl_rushing_predictor_{self.model_type}.pkl"
        
        model_data = {
            'model': self.model,
            'processor': self.processor,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
        return str(filepath)
    
    def load_model(self, filepath: str) -> None:
        #Load trained model and processor
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.processor = model_data['processor']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        #Calculate evaluation metrics
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def _add_confidence_intervals(self, results: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
        #Add confidence intervals to predictions (for supported models)
        # This is a simplified approach - in production you'd use proper uncertainty quantification
        if hasattr(self.model, 'estimators_'):
            # For ensemble methods, use prediction variance
            predictions_all = np.array([
                estimator.predict(X_scaled) 
                for estimator in self.model.estimators_.flatten()
            ])
            
            pred_std = np.std(predictions_all, axis=0)
            results['prediction_lower'] = (results['predicted_rushing_yards'] - 1.96 * pred_std).astype(int)
            results['prediction_upper'] = (results['predicted_rushing_yards'] + 1.96 * pred_std).astype(int)
            results['prediction_confidence'] = (1 / (1 + pred_std)).round(2)  # Simple confidence score
        
        return results
    
    def _assess_injury_risk(self, df: pd.DataFrame) -> pd.Series:
        #Assess injury risk based on player characteristics
        risk_factors = (
            df['age'] * 0.1 +  # Age factor
            df['injury_history'] * 0.3 +  # History factor
            (1 - df['games_percentage']) * 0.4 +  # Durability factor
            (df['attempts_per_game'] / 20) * 0.2  # Usage factor
        )
        
        # Normalize to 0-1 scale
        risk_normalized = (risk_factors - risk_factors.min()) / (risk_factors.max() - risk_factors.min())
        
        return risk_normalized.round(2)
    
    def analyze_predictions(self, results: pd.DataFrame) -> Dict[str, any]:
        #Analyze and summarize predictions
        analysis = {
            'predicted_leader': {
                'name': results.iloc[0]['player_name'],
                'team': results.iloc[0]['team'],
                'predicted_yards': int(results.iloc[0]['predicted_rushing_yards']),
                'age': int(results.iloc[0]['age'])
            },
            'top_5_summary': results.head(5)[['player_name', 'team', 'predicted_rushing_yards']].to_dict('records'),
            'prediction_stats': {
                'mean_predicted_yards': results['predicted_rushing_yards'].mean(),
                'median_predicted_yards': results['predicted_rushing_yards'].median(),
                'std_predicted_yards': results['predicted_rushing_yards'].std(),
                'min_predicted_yards': results['predicted_rushing_yards'].min(),
                'max_predicted_yards': results['predicted_rushing_yards'].max()
            },
            'age_analysis': {
                'youngest_top10': results.head(10)['age'].min(),
                'oldest_top10': results.head(10)['age'].max(),
                'avg_age_top10': results.head(10)['age'].mean()
            },
            'team_distribution': results.head(10)['team'].value_counts().to_dict()
        }
        
        return analysis
    
    def compare_with_previous_year(self, results: pd.DataFrame) -> pd.DataFrame:
        #Compare predictions with previous year performance
        if 'previous_year_yards' in results.columns:
            results['predicted_change'] = results['predicted_rushing_yards'] - results['previous_year_yards']
            results['predicted_change_pct'] = (results['predicted_change'] / results['previous_year_yards'] * 100).round(1)
            
            # Categorize changes
            results['trend_category'] = pd.cut(
                results['predicted_change_pct'],
                bins=[-float('inf'), -10, -5, 5, 10, float('inf')],
                labels=['Major Decline', 'Minor Decline', 'Stable', 'Minor Improvement', 'Major Improvement']
            )
        
        return results
    
    def get_model_summary(self) -> Dict[str, any]:
        #Get summary of trained model
        if not self.is_trained:
            return {'status': 'Not trained'}
        
        summary = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'n_features': len(self.processor.feature_columns),
            'feature_names': self.processor.feature_columns[:10],  # First 10 features
            'model_params': self.model.get_params()
        }
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.get_feature_importance().head(5)
            summary['top_5_features'] = importance.to_dict('records')
        
        return summary
    
    def update_predictions_with_weekly_stats(
        self,
        weekly_stats: pd.DataFrame,
        weeks: List[int] = [1, 2, 3],
        alpha: float = 0.6,
        season_games: int = 17
    ) -> pd.DataFrame:
        """
        Adjust existing model predictions using weekly game stats.

        - weekly_stats: DataFrame with ['player_name','week','rushing_yards'].
        - weeks: which weeks to use (default [1,2,3]).
        - alpha: weight given to model prediction vs observed pace.
        - season_games: games in season (default 17).

        Requires that self.predictions_df (or similar) contains
        'player_name' and 'predicted_rushing_yards'.
        """
        # Expect the model to have a predictions dataframe attribute
        if hasattr(self, 'predictions_df') and isinstance(self.predictions_df, pd.DataFrame):
            base_results = self.predictions_df.copy()
        elif hasattr(self, 'results') and isinstance(self.results, pd.DataFrame):
            base_results = self.results.copy()
        else:
            raise ValueError("No predictions found on the model instance. Run the prediction method first and ensure predictions_df or results exists.")

        adjusted = update_predictions_with_weekly_stats(
            results=base_results,
            weekly_stats=weekly_stats,
            weeks=weeks,
            alpha=alpha,
            season_games=season_games
        )

        # store adjusted predictions for later use
        self.adjusted_predictions_df = adjusted

        return adjusted
