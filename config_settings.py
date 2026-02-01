## NFL Rushing Predictor Configuration Settings

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Data settings - UPDATED for 2026 predictions
START_YEAR = 2000
END_YEAR = 2025  # Now includes 2025 data
CURRENT_SEASON = 2026  # Predicting 2026 season

# Model settings
MODEL_CONFIG = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': 42,
    'test_size': 0.2,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'subsample': 0.8
}

# Feature engineering settings
FEATURE_CONFIG = {
    'prime_age_range': (24, 28),
    'good_oline_threshold': 10,
    'bad_oline_threshold': 25,
    'high_volume_threshold': 250,
    'goal_line_td_rate': 0.08,
    'modern_era_start': 2018,
    'passing_era_start': 2010,
    'min_carries_threshold': 50  # Minimum carries to be included in dataset
}

# Team mappings for special cases (2026 season updates)
TEAM_OVERRIDES = {
    'Nick Chubb': 'CLE',  # Back to Cleveland for 2026
    'Saquon Barkley': 'PHI',
    'Derrick Henry': 'BAL',
    'Josh Jacobs': 'GB'
}

# Pro Football Reference settings
PFR_CONFIG = {
    'base_url': 'https://www.pro-football-reference.com',
    'request_delay': 1.0,  # Seconds between requests
    'max_retries': 3,
    'timeout': 30
}

# Prediction settings
PREDICTION_CONFIG = {
    'top_n_predictions': 10,
    'confidence_intervals': True,
    'include_injury_risk': True,
    'include_breakout_score': True,
    'min_prediction_threshold': 400  # Minimum predicted yards to be considered
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': LOGS_DIR / 'nfl_predictor.log'
}

# Current season player data template (2026 predictions)
# This includes 2025 actual performance data
CURRENT_PLAYERS_2026 = [
    {
        'player_name': 'Saquon Barkley',
        'team': 'PHI',
        'age': 29,
        'games_played': 17,
        'previous_year_yards': 2005,
        'previous_year_attempts': 345,
        'injury_history': 1,
        'offensive_line_rank': 12,
        'yards_per_carry': 5.8,
        'rushing_tds': 13
    },
    {
        'player_name': 'Derrick Henry',
        'team': 'BAL',
        'age': 32,
        'games_played': 17,
        'previous_year_yards': 1921,
        'previous_year_attempts': 325,
        'injury_history': 0,
        'offensive_line_rank': 6,
        'yards_per_carry': 5.9,
        'rushing_tds': 16
    },
    {
        'player_name': 'Josh Jacobs',
        'team': 'GB',
        'age': 28,
        'games_played': 17,
        'previous_year_yards': 1329,
        'previous_year_attempts': 277,
        'injury_history': 0,
        'offensive_line_rank': 15,
        'yards_per_carry': 4.8,
        'rushing_tds': 11
    },
    {
        'player_name': 'Christian McCaffrey',
        'team': 'SF',
        'age': 30,
        'games_played': 17,
        'previous_year_yards': 1459,
        'previous_year_attempts': 317,
        'injury_history': 2,
        'offensive_line_rank': 8,
        'yards_per_carry': 4.6,
        'rushing_tds': 14
    },
    {
        'player_name': 'Nick Chubb',
        'team': 'CLE',
        'age': 30,
        'games_played': 17,
        'previous_year_yards': 1127,
        'previous_year_attempts': 262,
        'injury_history': 2,
        'offensive_line_rank': 14,
        'yards_per_carry': 4.3,
        'rushing_tds': 7
    },
    {
        'player_name': 'Jonathan Taylor',
        'team': 'IND',
        'age': 27,
        'games_played': 17,
        'previous_year_yards': 1024,
        'previous_year_attempts': 233,
        'injury_history': 1,
        'offensive_line_rank': 18,
        'yards_per_carry': 4.4,
        'rushing_tds': 9
    },
    {
        'player_name': 'Kenneth Walker III',
        'team': 'SEA',
        'age': 26,
        'games_played': 17,
        'previous_year_yards': 905,
        'previous_year_attempts': 192,
        'injury_history': 0,
        'offensive_line_rank': 22,
        'yards_per_carry': 4.7,
        'rushing_tds': 8
    },
    {
        'player_name': 'Breece Hall',
        'team': 'NYJ',
        'age': 25,
        'games_played': 17,
        'previous_year_yards': 994,
        'previous_year_attempts': 221,
        'injury_history': 1,
        'offensive_line_rank': 20,
        'yards_per_carry': 4.5,
        'rushing_tds': 5
    },
    {
        'player_name': 'Bijan Robinson',
        'team': 'ATL',
        'age': 24,
        'games_played': 17,
        'previous_year_yards': 976,
        'previous_year_attempts': 213,
        'injury_history': 0,
        'offensive_line_rank': 16,
        'yards_per_carry': 4.6,
        'rushing_tds': 8
    },
    {
        'player_name': 'De\'Von Achane',
        'team': 'MIA',
        'age': 24,
        'games_played': 17,
        'previous_year_yards': 800,
        'previous_year_attempts': 157,
        'injury_history': 0,
        'offensive_line_rank': 19,
        'yards_per_carry': 5.1,
        'rushing_tds': 6
    },
    {
        'player_name': 'Jahmyr Gibbs',
        'team': 'DET',
        'age': 23,
        'games_played': 17,
        'previous_year_yards': 1024,
        'previous_year_attempts': 186,
        'injury_history': 0,
        'offensive_line_rank': 10,
        'yards_per_carry': 5.5,
        'rushing_tds': 12
    },
    {
        'player_name': 'Kyren Williams',
        'team': 'LAR',
        'age': 25,
        'games_played': 17,
        'previous_year_yards': 1144,
        'previous_year_attempts': 258,
        'injury_history': 0,
        'offensive_line_rank': 17,
        'yards_per_carry': 4.4,
        'rushing_tds': 12
    },
    {
        'player_name': 'James Cook',
        'team': 'BUF',
        'age': 26,
        'games_played': 17,
        'previous_year_yards': 1009,
        'previous_year_attempts': 207,
        'injury_history': 0,
        'offensive_line_rank': 13,
        'yards_per_carry': 4.9,
        'rushing_tds': 16
    },
    {
        'player_name': 'Rachaad White',
        'team': 'TB',
        'age': 26,
        'games_played': 17,
        'previous_year_yards': 990,
        'previous_year_attempts': 224,
        'injury_history': 0,
        'offensive_line_rank': 21,
        'yards_per_carry': 4.4,
        'rushing_tds': 7
    },
    {
        'player_name': 'David Montgomery',
        'team': 'DET',
        'age': 28,
        'games_played': 17,
        'previous_year_yards': 775,
        'previous_year_attempts': 185,
        'injury_history': 1,
        'offensive_line_rank': 10,
        'yards_per_carry': 4.2,
        'rushing_tds': 12
    }
]

# Future enhancement flags
FUTURE_ENHANCEMENTS = {
    'enable_injury_probability_model': False,
    'enable_real_time_updates': False,
    'enable_betting_odds_comparison': False,
    'enable_web_dashboard': False,
    'enable_multi_position_predictions': False,
    'enable_weekly_predictions': False,
    'enable_playoff_predictions': False,
    'enable_advanced_ensemble': False
}