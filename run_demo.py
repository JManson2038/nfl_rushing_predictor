"""Run a small demo: train on fallback historical data and predict current season leaders.

This script uses the project's existing modules without requiring package restructuring.
"""
import logging
from pathlib import Path
import importlib.util
import sys


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).parent

# Load config_settings as config (module name: config_settings)
config = load_module_from_path('config_settings', ROOT / 'config_settings.py')

# Load other modules
data_loader_mod = load_module_from_path('data_loader', ROOT / 'data_loader.py')
data_processor_mod = load_module_from_path('data_processor', ROOT / 'data_processor.py')
utils_mod = load_module_from_path('utils_helpers', ROOT / 'utils_helpers.py')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


def main():
    logger = utils_mod.setup_logging(log_level='INFO')

    loader = data_loader_mod.NFLDataLoader(use_cache=False)
    logger.info('Loading historical data (fallback)...')
    df_hist = loader.load_historical_data()

    processor = data_processor_mod.NFLDataProcessor()
    df_clean = processor.clean_data(df_hist)
    df_eng = processor.engineer_features(df_clean)

    # Prepare features and target
    X = processor.prepare_features(df_eng, fit_encoders=True)
    y = processor.create_target_variable(df_eng, target_col='rushing_yards')

    # Build model using config MODEL_CONFIG (filter estimator args)
    cfg = config.MODEL_CONFIG
    model_kwargs = {k: v for k, v in cfg.items() if k in {
        'n_estimators', 'learning_rate', 'max_depth', 'random_state', 'subsample',
        'min_samples_split', 'min_samples_leaf'
    }}

    # Small safety: if dataset too small, reduce estimators
    if len(X) < 50 and 'n_estimators' in model_kwargs:
        model_kwargs['n_estimators'] = min(50, model_kwargs['n_estimators'])

    model = GradientBoostingRegressor(**model_kwargs)

    # Split
    test_size = cfg.get('test_size', 0.2)
    rs = cfg.get('random_state', 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs)

    # Scale
    X_train_scaled = processor.scale_features(X_train, fit_scaler=True)
    X_test_scaled = processor.scale_features(X_test, fit_scaler=False)

    logger.info('Training model...')
    model.fit(X_train_scaled, y_train)
    logger.info('Model trained')

    # Load current season players
    df_current = loader.load_current_season_data()
    df_current_eng = processor.engineer_features(df_current)
    X_cur = processor.prepare_features(df_current_eng, fit_encoders=False)
    X_cur_scaled = processor.scale_features(X_cur, fit_scaler=False)

    preds = model.predict(X_cur_scaled)
    results = df_current.copy()
    results['predicted_rushing_yards'] = preds.astype(int)

    # Format and print
    output = utils_mod.format_predictions_output(results.sort_values('predicted_rushing_yards', ascending=False).reset_index(drop=True))
    print(output)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # Print a concise error so the user can act
        print(f'ERROR: {e}')
        raise
