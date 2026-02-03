"""NFL Data Processor
Handles feature engineering and data preprocessing for ML models with safety measures.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings

# Import configuration
from config_settings import FEATURE_CONFIG


class NFLDataProcessor:
    """Process and engineer features for NFL rushing prediction with robust error handling."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.logger = logging.getLogger(__name__)
        self._fitted = False
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for better predictions.
        
        Args:
            df: Input DataFrame with raw NFL data
            
        Returns:
            DataFrame with engineered features
            
        Raises:
            ValueError: If required columns are missing
        """
        self.logger.info("Engineering features...")
        
        if df is None or df.empty:
            self.logger.warning("Empty DataFrame provided to engineer_features")
            return df
        
        df = df.copy()
        
        # Validate required columns exist
        required_cols = ['age', 'rushing_yards', 'games_played', 'rushing_attempts']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            # Add missing columns with default values
            for col in missing_cols:
                df[col] = self._get_default_value(col)
        
        try:
            # Age-based features
            df['age_squared'] = df['age'] ** 2
            df['prime_age'] = (
                (df['age'] >= FEATURE_CONFIG['prime_age_range'][0]) & 
                (df['age'] <= FEATURE_CONFIG['prime_age_range'][1])
            ).astype(int)
            
            # Performance metrics - safe division
            df['yards_per_game'] = np.where(
                df['games_played'] > 0,
                df['rushing_yards'] / df['games_played'],
                0
            )
            df['attempts_per_game'] = np.where(
                df['games_played'] > 0,
                df['rushing_attempts'] / df['games_played'],
                0
            )
            
            if 'rushing_tds' in df.columns:
                df['tds_per_game'] = np.where(
                    df['games_played'] > 0,
                    df['rushing_tds'] / df['games_played'],
                    0
                )
            else:
                df['tds_per_game'] = 0
            
            # Team quality indicators
            if 'offensive_line_rank' in df.columns:
                df['good_oline'] = (df['offensive_line_rank'] <= FEATURE_CONFIG['good_oline_threshold']).astype(int)
                df['bad_oline'] = (df['offensive_line_rank'] >= FEATURE_CONFIG['bad_oline_threshold']).astype(int)
                df['oline_score'] = 33 - df['offensive_line_rank']  # Higher is better
            else:
                df['good_oline'] = 0
                df['bad_oline'] = 0
                df['oline_score'] = 17  # Middle of the pack
            
            # Usage and role indicators
            df['high_volume_back'] = (df['rushing_attempts'] >= FEATURE_CONFIG['high_volume_threshold']).astype(int)
            
            if 'rushing_tds' in df.columns:
                df['goal_line_back'] = np.where(
                    df['rushing_attempts'] > 0,
                    (df['rushing_tds'] / df['rushing_attempts'] > FEATURE_CONFIG['goal_line_td_rate']).astype(int),
                    0
                )
            else:
                df['goal_line_back'] = 0
            
            df['workhorse_back'] = (df['attempts_per_game'] >= 15).astype(int)
            
            # Era-based features
            if 'season' in df.columns:
                df['modern_era'] = (df['season'] >= FEATURE_CONFIG['modern_era_start']).astype(int)
                df['passing_era'] = (df['season'] >= FEATURE_CONFIG['passing_era_start']).astype(int)
                df['early_2000s'] = (df['season'] <= 2007).astype(int)
            else:
                # Assume modern era for prediction
                df['modern_era'] = 1
                df['passing_era'] = 1
                df['early_2000s'] = 0
            
            # Career progression features
            df['career_stage'] = self._determine_career_stage(df)
            df['breakout_candidate'] = self._identify_breakout_candidates(df)
            
            # Durability features
            if 'season' in df.columns:
                df['games_percentage'] = df.apply(
                    lambda row: row['games_played'] / (16 if row.get('season', 2025) < 2021 else 17) 
                    if row['games_played'] > 0 else 0,
                    axis=1
                )
            else:
                df['games_percentage'] = df['games_played'] / 17
            
            if 'injury_history' in df.columns:
                df['injury_prone'] = (df['injury_history'] >= 2).astype(int)
            else:
                df['injury_prone'] = 0
            
            df['iron_man'] = (df['games_percentage'] >= 0.94).astype(int)
            
            # Previous season performance
            if 'previous_year_yards' in df.columns:
                if 'previous_year_attempts' in df.columns:
                    df['prev_year_ypc'] = np.where(
                        df['previous_year_attempts'] > 0,
                        df['previous_year_yards'] / df['previous_year_attempts'],
                        df.get('yards_per_carry', 4.5)
                    )
                else:
                    # Estimate attempts from yards
                    df['prev_year_ypc'] = np.where(
                        df['rushing_attempts'] > 0,
                        df['previous_year_yards'] / df['rushing_attempts'],
                        df.get('yards_per_carry', 4.5)
                    )
                
                df['year_over_year_change'] = df['rushing_yards'] - df['previous_year_yards']
                df['trending_up'] = (df['year_over_year_change'] > 100).astype(int)
                df['trending_down'] = (df['year_over_year_change'] < -100).astype(int)
            else:
                df['prev_year_ypc'] = df.get('yards_per_carry', 4.5)
                df['year_over_year_change'] = 0
                df['trending_up'] = 0
                df['trending_down'] = 0
            
            # Efficiency metrics
            if 'yards_per_carry' in df.columns:
                df['efficiency_score'] = (df['yards_per_carry'] - 4.0) * df['rushing_attempts'] / 100
            else:
                df['efficiency_score'] = 0
            
            if 'rushing_tds' in df.columns:
                df['red_zone_efficiency'] = np.where(
                    df['rushing_attempts'] > 0,
                    df['rushing_tds'] / np.maximum(df['rushing_attempts'] / 20, 1),
                    0
                )
            else:
                df['red_zone_efficiency'] = 0
            
            # Clip extreme values to prevent outliers from breaking the model
            df = self._clip_extreme_values(df)
            
            self.logger.info(f"Engineered {len([col for col in df.columns if col not in ['player_name', 'season', 'team']])} features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            # Return original dataframe if feature engineering fails
            return df
    
    def _get_default_value(self, column: str) -> float:
        """Get default value for missing columns."""
        defaults = {
            'age': 25,
            'rushing_yards': 0,
            'games_played': 17,
            'rushing_attempts': 200,
            'rushing_tds': 5,
            'yards_per_carry': 4.5,
            'offensive_line_rank': 17,
            'injury_history': 0,
            'previous_year_yards': 800
        }
        return defaults.get(column, 0)
    
    def _clip_extreme_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip extreme values to prevent outliers."""
        try:
            # Define reasonable ranges for key features
            clip_config = {
                'age': (18, 40),
                'yards_per_game': (0, 200),
                'attempts_per_game': (0, 35),
                'yards_per_carry': (0, 15),
                'efficiency_score': (-50, 50),
            }
            
            for col, (min_val, max_val) in clip_config.items():
                if col in df.columns:
                    df[col] = df[col].clip(min_val, max_val)
            
            return df
        except Exception as e:
            self.logger.warning(f"Error clipping values: {e}")
            return df
    
    def _determine_career_stage(self, df: pd.DataFrame) -> pd.Series:
        """
        Determine career stage based on age and experience.
        
        Args:
            df: DataFrame with 'age' column
            
        Returns:
            Series with career stage (0=Rookie/Sophomore, 1=Prime, 2=Veteran)
        """
        def classify_stage(age):
            try:
                age = float(age)
                if age <= 24:
                    return 0  # Rookie/Sophomore
                elif age <= 27:
                    return 1  # Prime
                else:
                    return 2  # Veteran
            except (ValueError, TypeError):
                return 1  # Default to prime if age is invalid

        return df['age'].apply(classify_stage)

    def _identify_breakout_candidates(self, df: pd.DataFrame) -> pd.Series:
        """
        Identify potential breakout candidates.
        
        Simple heuristic: low previous production but improving.
        
        Args:
            df: DataFrame with yards data
            
        Returns:
            Series with breakout indicator (0 or 1)
        """
        try:
            prev = df.get('previous_year_yards', pd.Series(0, index=df.index)).fillna(0)
            current = df.get('rushing_yards', pd.Series(0, index=df.index)).fillna(0)
            
            # Avoid division by zero
            prev_safe = prev.replace(0, 1)
            
            breakout = (
                ((prev < 600) & (current > 800)) | 
                ((current / prev_safe) > 1.5)
            ).astype(int)
            
            return breakout
        except Exception as e:
            self.logger.warning(f"Error identifying breakout candidates: {e}")
            return pd.Series(0, index=df.index)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df is None or df.empty:
            self.logger.warning("Empty DataFrame provided to clean_data")
            return df
        
        df = df.copy()
        
        # Ensure season exists
        if 'season' not in df.columns:
            from config_settings import CURRENT_SEASON
            df['season'] = CURRENT_SEASON
            self.logger.info(f"Added season column with value {CURRENT_SEASON}")

        # Fill missing numeric fields with sensible defaults
        numeric_defaults = {
            'rushing_yards': 0,
            'rushing_attempts': 200,
            'rushing_tds': 0,
            'offensive_line_rank': 17,
            'injury_history': 0,
            'games_played': 17,
            'yards_per_carry': 4.5
        }

        for col, default_val in numeric_defaults.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                # Convert to numeric, coerce errors to NaN, then fill with default
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val)
        
        # Ensure previous_year_yards exists
        if 'previous_year_yards' not in df.columns:
            df['previous_year_yards'] = df['rushing_yards']
            self.logger.info("Created previous_year_yards from rushing_yards")
        
        # Remove any rows with all NaN values
        df = df.dropna(how='all')
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values with 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        self.logger.info(f"Cleaned data: {len(df)} records remaining")
        return df

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate data quality and completeness.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_records': len(df),
            'missing_required_columns': [],
            'warnings': [],
            'is_valid': True
        }
        
        if df is None or df.empty:
            validation_results['is_valid'] = False
            validation_results['warnings'].append("DataFrame is empty")
            return validation_results
        
        # Check required columns
        required = ['player_name', 'season']
        missing = [c for c in required if c not in df.columns]
        validation_results['missing_required_columns'] = missing
        
        if missing:
            validation_results['is_valid'] = False
            validation_results['warnings'].append(f"Missing required columns: {missing}")
        
        # Check for suspicious values
        if 'age' in df.columns:
            invalid_ages = df[(df['age'] < 18) | (df['age'] > 40)]
            if len(invalid_ages) > 0:
                validation_results['warnings'].append(f"Found {len(invalid_ages)} records with invalid ages")
        
        if 'rushing_yards' in df.columns:
            negative_yards = df[df['rushing_yards'] < 0]
            if len(negative_yards) > 0:
                validation_results['warnings'].append(f"Found {len(negative_yards)} records with negative yards")
        
        if 'games_played' in df.columns:
            invalid_games = df[(df['games_played'] < 0) | (df['games_played'] > 17)]
            if len(invalid_games) > 0:
                validation_results['warnings'].append(f"Found {len(invalid_games)} records with invalid games_played")
        
        return validation_results

    def prepare_features(self, df: pd.DataFrame, fit_encoders: bool = False) -> pd.DataFrame:
        """
        Select and prepare features for model input.
        
        Args:
            df: DataFrame with engineered features
            fit_encoders: Whether to fit label encoders (True for training data)
            
        Returns:
            DataFrame with selected features
        """
        # Define a safe list of features that may exist after engineering
        candidates = [
            'age', 'age_squared', 'prime_age', 'yards_per_game', 'attempts_per_game',
            'tds_per_game', 'oline_score', 'high_volume_back', 'goal_line_back',
            'workhorse_back', 'modern_era', 'passing_era', 'early_2000s', 'career_stage',
            'breakout_candidate', 'games_percentage', 'injury_prone', 'iron_man',
            'prev_year_ypc', 'year_over_year_change', 'efficiency_score', 'red_zone_efficiency',
            'offensive_line_rank', 'yards_per_carry', 'previous_year_yards', 'trending_up',
            'trending_down', 'good_oline', 'bad_oline'
        ]

        features = []
        for c in candidates:
            if c in df.columns:
                features.append(c)
            else:
                # Create default column if missing
                df[c] = 0
                features.append(c)

        self.feature_columns = features
        X = df[self.feature_columns].copy()
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        # Replace infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        self.logger.info(f"Prepared {len(self.feature_columns)} features for modeling")
        return X

    def create_target_variable(self, df: pd.DataFrame, target_col: str = 'rushing_yards') -> pd.Series:
        """
        Create target variable for training.
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            
        Returns:
            Series with target values
        """
        if target_col not in df.columns:
            self.logger.warning(f"Target column '{target_col}' not found. Using zeros.")
            return pd.Series(0, index=df.index)
        
        target = df[target_col].copy()
        
        # Convert to numeric
        target = pd.to_numeric(target, errors='coerce')
        
        # Fill NaN values with 0
        target = target.fillna(0)
        
        # Clip to reasonable range (no negative yards, max 3000)
        target = target.clip(0, 3000)
        
        return target.astype(float)

    def scale_features(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """
        Scale features using StandardScaler.
        
        Args:
            X: Features DataFrame
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            Scaled feature array
        """
        try:
            if fit_scaler:
                self.scaler.fit(X)
                self._fitted = True
                self.logger.info("Fitted scaler to training data")
            
            if not self._fitted:
                self.logger.warning("Scaler not fitted. Fitting now...")
                self.scaler.fit(X)
                self._fitted = True
            
            X_scaled = self.scaler.transform(X)
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error scaling features: {e}. Returning unscaled features.")
            return X.values

    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for importance analysis."""
        return self.feature_columns.copy()
    
    def save_processor_state(self, filepath: Path) -> None:
        """
        Save processor state (scaler, encoders, feature columns).
        
        Args:
            filepath: Path to save the processor state
        """
        try:
            import joblib
            state = {
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                '_fitted': self._fitted
            }
            joblib.dump(state, filepath)
            self.logger.info(f"Saved processor state to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save processor state: {e}")
    
    def load_processor_state(self, filepath: Path) -> None:
        """
        Load processor state from file.
        
        Args:
            filepath: Path to load the processor state from
        """
        try:
            import joblib
            state = joblib.load(filepath)
            self.scaler = state['scaler']
            self.label_encoders = state['label_encoders']
            self.feature_columns = state['feature_columns']
            self._fitted = state['_fitted']
            self.logger.info(f"Loaded processor state from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load processor state: {e}")
