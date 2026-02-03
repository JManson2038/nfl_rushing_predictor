"""NFL Rushing Predictor Model
Machine learning model for predicting NFL rushing yards with safety measures.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
import joblib

from data_processor import NFLDataProcessor
from config_settings import MODEL_CONFIG, TRAINING_CONFIG


class NFLRushingPredictor:
    """
    NFL Rushing Yards Predictor with multiple model types and blending strategies.
    """
    
    SUPPORTED_MODELS = ['gradient_boosting', 'random_forest', 'ridge', 'lasso']
    BLEND_METHODS = ['linear', 'power', 'sigmoid', 'none']
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model to use ('gradient_boosting', 'random_forest', 'ridge', 'lasso')
        
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type
        self.model = self._create_model()
        self.processor = NFLDataProcessor()
        self.logger = logging.getLogger(__name__)
        self.is_trained = False
        self.feature_importance = None
        
        # Blending configuration
        self.blend_method = 'sigmoid'  # Default blending method
        self.blend_exponent = 1.0  # For power/sigmoid blending
        
    def _create_model(self) -> Union[GradientBoostingRegressor, RandomForestRegressor, Ridge, Lasso]:
        """
        Create the appropriate model based on model_type.
        
        Returns:
            Instantiated sklearn model
        """
        if self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=MODEL_CONFIG.get('n_estimators', 300),
                learning_rate=MODEL_CONFIG.get('learning_rate', 0.1),
                max_depth=MODEL_CONFIG.get('max_depth', 6),
                random_state=MODEL_CONFIG.get('random_state', 42),
                loss='absolute_error',  # More robust to outliers
                subsample=0.8,  # Add some randomness for regularization
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=MODEL_CONFIG.get('n_estimators', 300),
                max_depth=MODEL_CONFIG.get('max_depth', 6),
                random_state=MODEL_CONFIG.get('random_state', 42),
                n_jobs=-1  # Use all CPU cores
            )
        elif self.model_type == 'ridge':
            return Ridge(
                alpha=1.0,
                random_state=MODEL_CONFIG.get('random_state', 42)
            )
        elif self.model_type == 'lasso':
            return Lasso(
                alpha=1.0,
                random_state=MODEL_CONFIG.get('random_state', 42),
                max_iter=10000
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'rushing_yards',
        validate: bool = True
    ) -> Dict[str, float]:
        """
        Train the model on historical data.
        
        Args:
            df: Training DataFrame
            target_col: Name of target column
            validate: Whether to perform validation
            
        Returns:
            Dictionary with training metrics
            
        Raises:
            ValueError: If training data is invalid
        """
        self.logger.info(f"Starting training with {len(df)} records using {self.model_type}")
        
        if df is None or df.empty:
            raise ValueError("Training data is empty")
        
        # Validate data
        validation_results = self.processor.validate_data(df)
        if not validation_results['is_valid']:
            self.logger.warning(f"Data validation warnings: {validation_results['warnings']}")
        
        try:
            # Clean and prepare data
            df_clean = self.processor.clean_data(df)
            df_features = self.processor.engineer_features(df_clean)
            
            # Prepare features and target
            X = self.processor.prepare_features(df_features, fit_encoders=True)
            y = self.processor.create_target_variable(df_features, target_col)
            
            # Check for sufficient data
            if len(X) < 50:
                self.logger.warning(f"Only {len(X)} training samples. Results may be unreliable.")
            
            # Scale features
            X_scaled = self.processor.scale_features(X, fit_scaler=True)
            
            # Train/test split if validation is requested
            if validate and len(X) > 100:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y,
                    test_size=TRAINING_CONFIG.get('test_size', 0.2),
                    random_state=TRAINING_CONFIG.get('random_state', 42)
                )
            else:
                X_train, y_train = X_scaled, y
                X_test, y_test = None, None
            
            # Train the model
            self.logger.info("Training model...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(X_train, y_train)
            
            self.is_trained = True
            
            # Calculate metrics
            metrics = self._calculate_metrics(X_train, y_train, X_test, y_test)
            
            # Calculate feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': self.processor.get_feature_importance_names(),
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.logger.info(f"Top 5 features: {list(self.feature_importance.head()['feature'])}")
            
            self.logger.info(f"Training complete. MAE: {metrics.get('mae', 'N/A'):.2f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
    
    def _calculate_metrics(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate model performance metrics.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
            
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        try:
            # Training metrics
            y_train_pred = self.model.predict(X_train)
            metrics['train_mae'] = mean_absolute_error(y_train, y_train_pred)
            metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
            metrics['train_r2'] = r2_score(y_train, y_train_pred)
            
            # Test metrics if available
            if X_test is not None and y_test is not None:
                y_test_pred = self.model.predict(X_test)
                metrics['test_mae'] = mean_absolute_error(y_test, y_test_pred)
                metrics['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
                metrics['test_r2'] = r2_score(y_test, y_test_pred)
                metrics['mae'] = metrics['test_mae']  # Use test MAE as primary metric
            else:
                metrics['mae'] = metrics['train_mae']
            
            # Cross-validation score if we have enough data
            if len(X_train) > 100:
                try:
                    cv_scores = cross_val_score(
                        self.model, X_train, y_train, 
                        cv=min(5, len(X_train) // 20),
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1
                    )
                    metrics['cv_mae_mean'] = -cv_scores.mean()
                    metrics['cv_mae_std'] = cv_scores.std()
                except Exception as e:
                    self.logger.warning(f"Cross-validation failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def predict(
        self, 
        df: pd.DataFrame,
        sort_by: str = 'confidence',
        return_raw: bool = False
    ) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with player data to predict
            sort_by: How to sort results ('confidence', 'predicted_yards', 'player_name')
            return_raw: If True, return raw model predictions without blending
            
        Returns:
            DataFrame with predictions and metadata
            
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        if df is None or df.empty:
            self.logger.warning("Empty DataFrame provided for prediction")
            return pd.DataFrame()
        
        self.logger.info(f"Making predictions for {len(df)} players")
        
        try:
            # Store original data
            df_original = df.copy()
            
            # Clean and prepare data
            df_clean = self.processor.clean_data(df)
            df_features = self.processor.engineer_features(df_clean)
            
            # Prepare features
            X = self.processor.prepare_features(df_features, fit_encoders=False)
            X_scaled = self.processor.scale_features(X, fit_scaler=False)
            
            # Make raw predictions
            raw_predictions = self.model.predict(X_scaled)
            raw_predictions = np.clip(raw_predictions, 0, 3000)  # Clip to reasonable range
            
            # Create results DataFrame
            results = df_original.copy()
            results['raw_model_prediction'] = raw_predictions
            
            # Apply blending with previous year if available
            if not return_raw and 'previous_year_yards' in results.columns:
                adjusted_predictions = self._blend_predictions(
                    raw_predictions,
                    results['previous_year_yards'].values
                )
                results['predicted_rushing_yards'] = adjusted_predictions
            else:
                results['predicted_rushing_yards'] = raw_predictions
            
            # Add confidence scores
            results['prediction_confidence'] = self._calculate_confidence(
                results, df_features
            )
            
            # Calculate predicted change from previous year
            if 'previous_year_yards' in results.columns:
                results['predicted_change'] = (
                    results['predicted_rushing_yards'] - results['previous_year_yards']
                )
                results['predicted_change_pct'] = (
                    results['predicted_change'] / results['previous_year_yards'].replace(0, 1) * 100
                ).round(1)
            
            # Add injury risk if injury_history is available
            if 'injury_history' in results.columns:
                results['injury_risk'] = self._calculate_injury_risk(results)
            
            # Sort results
            results = self._sort_predictions(results, sort_by)
            
            # Add rank
            results['prediction_rank'] = range(1, len(results) + 1)
            
            self.logger.info(f"Predictions complete. Top prediction: {results.iloc[0]['player_name']} - {results.iloc[0]['predicted_rushing_yards']:.0f} yards")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            raise
    
    def _blend_predictions(
        self,
        model_predictions: np.ndarray,
        previous_yards: np.ndarray
    ) -> np.ndarray:
        """
        Blend model predictions with previous year performance.
        
        Args:
            model_predictions: Raw model predictions
            previous_yards: Previous year rushing yards
            
        Returns:
            Blended predictions
        """
        if self.blend_method == 'none':
            return model_predictions
        
        # Clean inputs
        model_predictions = np.nan_to_num(model_predictions, nan=0.0, posinf=3000, neginf=0)
        previous_yards = np.nan_to_num(previous_yards, nan=0.0, posinf=3000, neginf=0)
        
        if self.blend_method == 'linear':
            # Simple weighted average
            alpha = 0.7  # Weight for model prediction
            blended = alpha * model_predictions + (1 - alpha) * previous_yards
            
        elif self.blend_method == 'power':
            # Use power function for blending
            diff = model_predictions - previous_yards
            adjustment = np.sign(diff) * np.abs(diff) ** self.blend_exponent
            blended = previous_yards + adjustment
            
        elif self.blend_method == 'sigmoid':
            # Sigmoid blending - smooth transition
            diff = model_predictions - previous_yards
            # Scale by exponent (controls steepness)
            scaled_diff = diff / (500 / self.blend_exponent)
            sigmoid = 1 / (1 + np.exp(-scaled_diff))
            blended = previous_yards + diff * sigmoid
            
        else:
            blended = model_predictions
        
        # Ensure reasonable values
        blended = np.clip(blended, 0, 3000)
        
        return blended
    
    def _calculate_confidence(
        self,
        results: pd.DataFrame,
        features: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate confidence scores for predictions.
        
        Args:
            results: Results DataFrame
            features: Features used for prediction
            
        Returns:
            Array of confidence scores (0-1)
        """
        try:
            confidence = np.ones(len(results))
            
            # Reduce confidence for players with injury history
            if 'injury_history' in results.columns:
                injury_penalty = results['injury_history'].values * 0.1
                confidence -= injury_penalty
            
            # Reduce confidence for older players
            if 'age' in results.columns:
                age_penalty = np.maximum(0, (results['age'].values - 30) * 0.05)
                confidence -= age_penalty
            
            # Reduce confidence for players with bad o-line
            if 'offensive_line_rank' in results.columns:
                oline_penalty = np.maximum(0, (results['offensive_line_rank'].values - 20) * 0.01)
                confidence -= oline_penalty
            
            # Boost confidence for players in prime age
            if 'age' in results.columns:
                prime_mask = (results['age'].values >= 24) & (results['age'].values <= 28)
                confidence[prime_mask] += 0.1
            
            # Clip to 0-1 range
            confidence = np.clip(confidence, 0.1, 1.0)
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"Error calculating confidence: {e}")
            return np.ones(len(results)) * 0.5
    
    def _calculate_injury_risk(self, results: pd.DataFrame) -> np.ndarray:
        """
        Calculate injury risk scores.
        
        Args:
            results: Results DataFrame
            
        Returns:
            Array of injury risk scores (0-1)
        """
        try:
            risk = np.zeros(len(results))
            
            # Base risk from injury history
            if 'injury_history' in results.columns:
                risk += results['injury_history'].values * 0.2
            
            # Age increases risk
            if 'age' in results.columns:
                age_risk = np.maximum(0, (results['age'].values - 28) * 0.05)
                risk += age_risk
            
            # High usage increases risk
            if 'previous_year_yards' in results.columns:
                usage_risk = (results['previous_year_yards'].values / 2500) * 0.2
                risk += usage_risk
            
            # Clip to 0-1 range
            risk = np.clip(risk, 0, 1)
            
            return risk
            
        except Exception as e:
            self.logger.warning(f"Error calculating injury risk: {e}")
            return np.zeros(len(results))
    
    def _sort_predictions(
        self,
        results: pd.DataFrame,
        sort_by: str
    ) -> pd.DataFrame:
        """
        Sort prediction results.
        
        Args:
            results: Results DataFrame
            sort_by: Column to sort by
            
        Returns:
            Sorted DataFrame
        """
        sort_column_map = {
            'confidence': 'prediction_confidence',
            'predicted_yards': 'predicted_rushing_yards',
            'player_name': 'player_name'
        }
        
        sort_col = sort_column_map.get(sort_by, 'predicted_rushing_yards')
        ascending = True if sort_by == 'player_name' else False
        
        if sort_col in results.columns:
            return results.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        else:
            self.logger.warning(f"Sort column '{sort_col}' not found. Sorting by predicted_rushing_yards.")
            return results.sort_values('predicted_rushing_yards', ascending=False).reset_index(drop=True)
    
    def apply_top_n_uplift(
        self,
        results: pd.DataFrame,
        top_n: int = 3,
        multiplier: float = 1.1
    ) -> pd.DataFrame:
        """
        Apply an uplift multiplier to top N predictions based on raw model confidence.
        
        Args:
            results: Results DataFrame
            top_n: Number of top players to uplift
            multiplier: Multiplier to apply
            
        Returns:
            Updated results DataFrame
        """
        if 'raw_model_prediction' not in results.columns:
            self.logger.warning("raw_model_prediction not found. Cannot apply top-N uplift.")
            return results
        
        results = results.copy()
        
        # Get top N by raw model prediction
        top_indices = results.nlargest(top_n, 'raw_model_prediction').index
        
        # Apply multiplier
        results.loc[top_indices, 'predicted_rushing_yards'] *= multiplier
        results.loc[top_indices, 'predicted_rushing_yards'] = results.loc[top_indices, 'predicted_rushing_yards'].round()
        
        # Recalculate predicted change
        if 'previous_year_yards' in results.columns:
            results['predicted_change'] = (
                results['predicted_rushing_yards'] - results['previous_year_yards']
            )
            results['predicted_change_pct'] = (
                results['predicted_change'] / results['previous_year_yards'].replace(0, 1) * 100
            ).round(1)
        
        # Re-sort
        results = results.sort_values('predicted_rushing_yards', ascending=False).reset_index(drop=True)
        results['prediction_rank'] = range(1, len(results) + 1)
        
        self.logger.info(f"Applied {multiplier}x uplift to top {top_n} players")
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            self.logger.warning("Feature importance not available for this model type")
            return pd.DataFrame()
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath: Path) -> None:
        """
        Save model and processor to disk.
        
        Args:
            filepath: Path to save the model
        """
        try:
            model_data = {
                'model': self.model,
                'processor': self.processor,
                'model_type': self.model_type,
                'is_trained': self.is_trained,
                'feature_importance': self.feature_importance,
                'blend_method': self.blend_method,
                'blend_exponent': self.blend_exponent
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, filepath: Path) -> None:
        """
        Load model and processor from disk.
        
        Args:
            filepath: Path to load the model from
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.processor = model_data['processor']
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']
            self.feature_importance = model_data.get('feature_importance')
            self.blend_method = model_data.get('blend_method', 'sigmoid')
            self.blend_exponent = model_data.get('blend_exponent', 1.0)
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
