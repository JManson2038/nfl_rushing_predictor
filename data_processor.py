# NFL Data Processor
# Handles feature engineering and data preprocessing for ML models

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config_settings import FEATURE_CONFIG

class NFLDataProcessor:
    """Process and engineer features for NFL rushing prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.logger = logging.getLogger(__name__)
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate input data"""
        self.logger.info("Cleaning data...")
        
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['player_name', 'season'], keep='first')
        
        # Remove records with missing critical values
        critical_cols = ['player_name', 'season', 'rushing_yards', 'games_played']
        df = df.dropna(subset=critical_cols)
        
        # Fill missing values for optional columns
        if 'rushing_attempts' not in df.columns:
            df['rushing_attempts'] = (df['rushing_yards'] / 4.5).astype(int)
        
        if 'rushing_tds' not in df.columns:
            df['rushing_tds'] = (df['rushing_yards'] / 150).astype(int)
            
        if 'yards_per_carry' not in df.columns:
            df['yards_per_carry'] = df['rushing_yards'] / df['rushing_attempts']
            
        if 'offensive_line_rank' not in df.columns:
            df['offensive_line_rank'] = 16  # League average
            
        if 'injury_history' not in df.columns:
            df['injury_history'] = 0
            
        if 'previous_year_yards' not in df.columns:
            df['previous_year_yards'] = df['rushing_yards']
            
        if 'previous_year_attempts' not in df.columns:
            df['previous_year_attempts'] = df['rushing_attempts']
        
        # Ensure positive values
        df['rushing_yards'] = df['rushing_yards'].clip(lower=0)
        df['rushing_attempts'] = df['rushing_attempts'].clip(lower=1)
        df['games_played'] = df['games_played'].clip(lower=1, upper=17)
        df['age'] = df['age'].clip(lower=20, upper=40)
        
        # Fix invalid yards per carry
        df['yards_per_carry'] = np.where(
            df['yards_per_carry'] > 15,  # Unrealistic YPC
            df['rushing_yards'] / df['rushing_attempts'],
            df['yards_per_carry']
        )
        
        self.logger.info(f"Cleaned data: {len(df)} records remaining")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """Validate data quality and return statistics"""
        validation = {
            'total_records': len(df),
            'unique_players': df['player_name'].nunique(),
            'seasons_covered': f"{df['season'].min()}-{df['season'].max()}",
            'avg_yards': df['rushing_yards'].mean(),
            'avg_attempts': df['rushing_attempts'].mean(),
            'missing_values': df.isnull().sum().to_dict()
        }
        return validation
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for better predictions"""
        self.logger.info("Engineering features...")
        
        df = df.copy()
        
        # Age-based features
        df['age_squared'] = df['age'] ** 2
        df['prime_age'] = (
            (df['age'] >= FEATURE_CONFIG['prime_age_range'][0]) & 
            (df['age'] <= FEATURE_CONFIG['prime_age_range'][1])
        ).astype(int)
        
        # Performance metrics
        df['yards_per_game'] = df['rushing_yards'] / df['games_played']
        df['attempts_per_game'] = df['rushing_attempts'] / df['games_played']
        df['tds_per_game'] = df['rushing_tds'] / df['games_played']
        
        # Team quality indicators
        df['good_oline'] = (df['offensive_line_rank'] <= FEATURE_CONFIG['good_oline_threshold']).astype(int)
        df['bad_oline'] = (df['offensive_line_rank'] >= FEATURE_CONFIG['bad_oline_threshold']).astype(int)
        df['oline_score'] = 33 - df['offensive_line_rank']  # Higher is better
        
        # Usage and role indicators
        df['high_volume_back'] = (df['rushing_attempts'] >= FEATURE_CONFIG['high_volume_threshold']).astype(int)
        df['goal_line_back'] = (df['rushing_tds'] / df['rushing_attempts'] > FEATURE_CONFIG['goal_line_td_rate']).astype(int)
        df['workhorse_back'] = (df['attempts_per_game'] >= 15).astype(int)
        
        # Era-based features
        if 'season' in df.columns:
            df['modern_era'] = (df['season'] >= FEATURE_CONFIG['modern_era_start']).astype(int)
            df['passing_era'] = (df['season'] >= FEATURE_CONFIG['passing_era_start']).astype(int)
            df['early_2000s'] = (df['season'] <= 2007).astype(int)

        # Career progression features
        df['career_stage'] = self._determine_career_stage(df)

        # Previous season performance
        df['prev_year_ypc'] = df['previous_year_yards'] / df.get('previous_year_attempts', df['rushing_attempts'])
        df['year_over_year_change'] = df['rushing_yards'] - df['previous_year_yards']
        df['trending_up'] = (df['year_over_year_change'] > 100).astype(int)
        df['trending_down'] = (df['year_over_year_change'] < -100).astype(int)

        # Identify breakout candidates after trending_up is available
        df['breakout_candidate'] = self._identify_breakout_candidates(df)

        # Durability features
        df['games_percentage'] = df.apply(
            lambda row: row['games_played'] / (16 if row.get('season', 2025) < 2021 else 17),
            axis=1
        )
        df['injury_prone'] = (df['injury_history'] >= 2).astype(int)
        df['iron_man'] = (df['games_percentage'] >= 0.94).astype(int)

        # Efficiency metrics
        df['efficiency_score'] = (df['yards_per_carry'] - 4.0) * df['rushing_attempts'] / 100
        df['red_zone_efficiency'] = df['rushing_tds'] / np.maximum(df['rushing_attempts'] / 20, 1)

        self.logger.info(f"Engineered {len([col for col in df.columns if col not in ['player_name', 'season', 'team']])} features")
        return df
    
    def _determine_career_stage(self, df: pd.DataFrame) -> pd.Series:
        """Determine career stage based on age and experience"""
        def classify_stage(age):
            if age <= 24:
                return 0  # Rookie/Sophomore
            elif age <= 27:
                return 1  # Prime
            elif age <= 30:
                return 2  # Veteran
            else:
                return 3  # Declining
        
        return df['age'].apply(classify_stage)
    
    def _identify_breakout_candidates(self, df: pd.DataFrame) -> pd.Series:
        """Identify potential breakout candidates"""
        # Breakout criteria: young, trending up, increasing usage
        breakout = (
            (df['age'] <= 25) &
            (df['trending_up'] == 1) &
            (df['attempts_per_game'] >= 12)
        ).astype(int)
        
        return breakout
    
    def prepare_features(self, df: pd.DataFrame, fit_encoders: bool = False) -> pd.DataFrame:
        """Prepare features for model training/prediction"""
        self.logger.info("Preparing features for model...")
        
        # Select numeric features for model
        exclude_cols = ['player_name', 'team', 'season']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Remove target variable if present
        if 'rushing_yards' in feature_cols:
            feature_cols.remove('rushing_yards')
        
        # If we're preparing features for inference and have stored feature names
        # from training, use that exact ordering and add any missing columns with zeros.
        if not fit_encoders and self.feature_columns:
            expected = self.feature_columns
            # Create a DataFrame with expected columns
            X = pd.DataFrame(index=df.index)
            for col in expected:
                if col in df.columns:
                    X[col] = df[col]
                else:
                    # Missing feature: fill with zeros
                    X[col] = 0
        else:
            X = df[feature_cols].copy()

        # Handle any remaining missing values
        X = X.fillna(X.mean())

        # Store feature columns for later use (during training)
        if fit_encoders:
            self.feature_columns = feature_cols
        
        self.logger.info(f"Prepared {len(feature_cols)} features")
        return X
    
    def scale_features(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Scale features using StandardScaler"""
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            self.logger.info("Fitted and transformed features")
        else:
            X_scaled = self.scaler.transform(X)
            self.logger.info("Transformed features using existing scaler")
        
        return X_scaled
    
    def create_target_variable(self, df: pd.DataFrame, target_col: str = 'rushing_yards') -> pd.Series:
        """Create target variable for training"""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        return df[target_col]
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for importance analysis"""
        return self.feature_columns