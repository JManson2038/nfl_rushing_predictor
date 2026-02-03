"""NFL Rushing Predictor Configuration Settings

This module contains all configuration settings for the NFL Rushing Predictor.
Sensitive data should be stored in environment variables, not here.
"""

import os
from pathlib import Path
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory {dir_path}: {e}")

# Data settings
START_YEAR = 2000
END_YEAR = 2024
CURRENT_SEASON = 2025

# Model settings
MODEL_CONFIG = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': 42,
}

# Training configuration separate from model hyperparameters
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42
}

# Feature engineering settings
FEATURE_CONFIG = {
    'prime_age_range': (24, 28),
    'good_oline_threshold': 10,
    'bad_oline_threshold': 25,
    'high_volume_threshold': 250,
    'goal_line_td_rate': 0.08,
    'modern_era_start': 2018,
    'passing_era_start': 2010
}

# Team mappings for special cases
TEAM_OVERRIDES = {
    'Nick Chubb': 'HOU'  # Special case: Nick Chubb to Houston
}

# Pro Football Reference settings - SAFE DEFAULTS
PFR_CONFIG = {
    'base_url': 'https://www.pro-football-reference.com',
    'request_delay': 3.0,  # Increased to 3 seconds - be respectful!
    'max_retries': 2,  # Reduced from 3
    'timeout': 10,  # Reduced from 30 to fail faster
    'user_agent': 'NFLRushingPredictor/1.0 (Educational Project)',
    'respect_robots_txt': True,
}

# API rate limiting settings
API_RATE_LIMITS = {
    'requests_per_minute': 10,  # Conservative limit
    'requests_per_hour': 100,
    'backoff_factor': 2.0,  # Exponential backoff multiplier
}

# Prediction settings
PREDICTION_CONFIG = {
    'top_n_predictions': 10,
    'confidence_intervals': True,
    'include_injury_risk': True
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': LOGS_DIR / 'nfl_predictor.log',
    'max_bytes': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5,
}

# Environment-based API keys (NEVER hardcode these!)
def get_api_key(service: str) -> Optional[str]:
    """
    Safely retrieve API keys from environment variables.
    
    Args:
        service: Name of the service (e.g., 'NFL_API', 'PRO_FOOTBALL_REF')
    
    Returns:
        API key if found, None otherwise
    """
    env_var_name = f"{service.upper()}_API_KEY"
    api_key = os.getenv(env_var_name)
    
    if api_key:
        # Never log the actual key!
        print(f"✓ Loaded API key for {service}")
    else:
        print(f"⚠ No API key found for {service} (set {env_var_name} env variable)")
    
    return api_key

# Current season player data template
CURRENT_PLAYERS_2025 = [
    {
        'player_name': 'Saquon Barkley',
        'team': 'PHI',
        'age': 28,
        'games_played': 17,
        'previous_year_yards': 2005,
        'injury_history': 1,
        'offensive_line_rank': 12,
        'yards_per_carry': 5.8
    },
    {
        'player_name': 'Derrick Henry',
        'team': 'BAL',
        'age': 31,
        'games_played': 17,
        'previous_year_yards': 1921,
        'injury_history': 0,
        'offensive_line_rank': 6,
        'yards_per_carry': 5.9
    },
    {
        'player_name': 'Josh Jacobs',
        'team': 'GB',
        'age': 27,
        'games_played': 17,
        'previous_year_yards': 1329,
        'injury_history': 0,
        'offensive_line_rank': 15,
        'yards_per_carry': 4.8
    },
    {
        'player_name': 'Christian McCaffrey',
        'team': 'SF',
        'age': 29,
        'games_played': 17,
        'previous_year_yards': 1459,
        'injury_history': 2,
        'offensive_line_rank': 8,
        'yards_per_carry': 4.6
    },
    {
        'player_name': 'Nick Chubb',
        'team': 'HOU',
        'age': 29,
        'games_played': 17,
        'previous_year_yards': 1127,
        'injury_history': 2,
        'offensive_line_rank': 14,
        'yards_per_carry': 4.3
    },
    {
        'player_name': 'Jonathan Taylor',
        'team': 'IND',
        'age': 26,
        'games_played': 17,
        'previous_year_yards': 1024,
        'injury_history': 1,
        'offensive_line_rank': 18,
        'yards_per_carry': 4.4
    },
    {
        'player_name': 'Kenneth Walker III',
        'team': 'SEA',
        'age': 25,
        'games_played': 17,
        'previous_year_yards': 905,
        'injury_history': 0,
        'offensive_line_rank': 22,
        'yards_per_carry': 4.7
    },
    {
        'player_name': 'Breece Hall',
        'team': 'NYJ',
        'age': 24,
        'games_played': 17,
        'previous_year_yards': 994,
        'injury_history': 1,
        'offensive_line_rank': 20,
        'yards_per_carry': 4.5
    },
    {
        'player_name': 'Bijan Robinson',
        'team': 'ATL',
        'age': 23,
        'games_played': 17,
        'previous_year_yards': 976,
        'injury_history': 0,
        'offensive_line_rank': 16,
        'yards_per_carry': 4.6
    },
    {
        'player_name': 'De\'Von Achane',
        'team': 'MIA',
        'age': 23,
        'games_played': 17,
        'previous_year_yards': 800,
        'injury_history': 0,
        'offensive_line_rank': 19,
        'yards_per_carry': 5.1
    }
]
