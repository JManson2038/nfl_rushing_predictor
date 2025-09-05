"""
NFL Data Processor
Handles feature engineering and data preprocessing for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from typing import Dict, List, Tuple
from pathlib import Path

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import FEATURE_CONFIG

class NFLDataProcessor:
    """Process and engineer features for NFL rushing prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.logger = logging.getLogger(__name__)
        
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
        df['modern_era'] = (df['season'] >= FEATURE_CONFIG['modern_era_start']).astype(int)
        df['passing_era'] = (df['season'] >= FEATURE_CONFIG['passing_era_start']).astype(int)
        df['early_2000s'] = (df['season'] <= 2007).astype(int)
        
        # Career progression features
        df['career_stage'] = self._determine_career_stage(df)
        df['breakout_candidate'] = self._identify_breakout_candidates(df)
        
        # Durability features
        df['games_percentage'] = df.apply(
            lambda row: row['games_played'] / (16 if row['season'] < 2021 else 17), 
            axis=1
        )
        df['injury_prone'] = (df['injury_history'] >= 2).astype(int)
        df['iron_man'] = (df['games_percentage'] >= 0.94).astype(int)
        
        # Previous season performance
        df['prev_year_ypc'] = df['previous_year_yards'] / df.get('previous_year_attempts', df['rushing_attempts'])
        df['year_over_year_change'] = df['rushing_yards'] - df['previous_year_yards']
        df['trending_up'] = (df['year_over_year_change'] > 100).astype(int)
        df['trending_down'] = (df['year_over_year_change'] < -100).astype(int)
        
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
            elif age <= 30