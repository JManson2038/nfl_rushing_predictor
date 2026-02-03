"""Run script for NFL Rushing Predictor

This script loads current season players, trains a model on historical data,
makes predictions, and prints formatted results with comprehensive error handling.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config_settings import LOGGING_CONFIG, MODELS_DIR
from data_loader import NFLDataLoader
from predictor_model import NFLRushingPredictor
from utils_helpers import (
    setup_logging, 
    format_predictions_output,
    export_predictions_to_csv,
    validate_dataframe
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run NFL Rushing Predictor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        '--model-type',
        choices=['gradient_boosting', 'random_forest', 'ridge', 'lasso'],
        default='gradient_boosting',
        help='Type of model to use'
    )
    
    # Blending options
    parser.add_argument(
        '--blend-method',
        choices=['linear', 'power', 'sigmoid', 'none'],
        default='sigmoid',
        help='Blending method for combining model and previous year'
    )
    parser.add_argument(
        '--blend-exponent',
        type=float,
        default=1.0,
        help='Exponent/steepness used for blending (power or sigmoid methods)'
    )
    
    # Top-N uplift options
    parser.add_argument(
        '--top-n',
        type=int,
        default=3,
        help='Number of top raw model players to uplift'
    )
    parser.add_argument(
        '--top-n-multiplier',
        type=float,
        default=1.1,
        help='Multiplier applied to top-N adjusted predictions'
    )
    
    # Data options
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable data caching (fetch fresh data)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=100,
        help='Minimum number of training samples required'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path.cwd(),
        help='Directory to save predictions'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='predictions_2025.csv',
        help='Filename for predictions CSV'
    )
    parser.add_argument(
        '--display-top-n',
        type=int,
        default=10,
        help='Number of top predictions to display'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save trained model to disk'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output (sets log level to DEBUG)'
    )
    
    return parser.parse_args()


def load_data(args: argparse.Namespace, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load historical and current season data.
    
    Args:
        args: Parsed command line arguments
        logger: Logger instance
        
    Returns:
        Tuple of (historical_df, current_df)
        
    Raises:
        RuntimeError: If data loading fails
    """
    logger.info("Loading data...")
    
    try:
        loader = NFLDataLoader(use_cache=not args.no_cache)
        
        # Load historical data for training
        hist_df = loader.load_historical_data()
        logger.info(f"Loaded {len(hist_df)} historical records")
        
        # Validate historical data
        is_valid, errors = validate_dataframe(
            hist_df,
            required_columns=['player_name', 'season', 'rushing_yards'],
            name="Historical data"
        )
        
        if not is_valid:
            logger.error(f"Historical data validation failed: {errors}")
            raise RuntimeError("Invalid historical data")
        
        # Check minimum samples
        if len(hist_df) < args.min_samples:
            logger.warning(
                f"Only {len(hist_df)} training samples available "
                f"(minimum recommended: {args.min_samples})"
            )
        
        # Load current players to predict
        current_df = loader.load_current_season_data()
        logger.info(f"Loaded {len(current_df)} current season players")
        
        # Validate current data
        is_valid, errors = validate_dataframe(
            current_df,
            required_columns=['player_name'],
            name="Current season data"
        )
        
        if not is_valid:
            logger.error(f"Current season data validation failed: {errors}")
            raise RuntimeError("Invalid current season data")
        
        # Display data summary
        summary = loader.get_data_summary(hist_df)
        logger.info(f"Data summary: {summary.get('unique_players', 0)} unique players")
        logger.info(f"Seasons covered: {summary.get('seasons_covered', 'N/A')}")
        
        return hist_df, current_df
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load data: {e}")


def train_model(
    args: argparse.Namespace,
    logger: logging.Logger,
    hist_df: pd.DataFrame
) -> NFLRushingPredictor:
    """
    Create and train the prediction model.
    
    Args:
        args: Parsed command line arguments
        logger: Logger instance
        hist_df: Historical training data
        
    Returns:
        Trained predictor instance
        
    Raises:
        RuntimeError: If training fails
    """
    logger.info(f"Creating {args.model_type} model...")
    
    try:
        # Create predictor
        predictor = NFLRushingPredictor(model_type=args.model_type)
        
        # Apply blending preferences from CLI
        predictor.blend_method = args.blend_method
        predictor.blend_exponent = args.blend_exponent
        
        logger.info(
            f"Blending configuration: method={args.blend_method}, "
            f"exponent={args.blend_exponent}"
        )
        
        # Train the model
        logger.info("Training model...")
        metrics = predictor.train(hist_df, target_col='rushing_yards', validate=True)
        
        # Log training results
        logger.info("Training completed successfully!")
        logger.info(f"Model performance metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Display feature importance if available
        feature_importance = predictor.get_feature_importance(top_n=10)
        if not feature_importance.empty:
            logger.info("Top 10 most important features:")
            for idx, row in feature_importance.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return predictor
        
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        
        # Try training on a smaller sample as fallback
        logger.warning("Attempting fallback training on reduced dataset...")
        try:
            sample_size = min(500, len(hist_df))
            sample = hist_df.sample(sample_size, random_state=42)
            
            predictor = NFLRushingPredictor(model_type=args.model_type)
            predictor.blend_method = args.blend_method
            predictor.blend_exponent = args.blend_exponent
            
            metrics = predictor.train(sample, target_col='rushing_yards', validate=False)
            logger.warning(f"Fallback training succeeded with {sample_size} samples")
            logger.warning("Predictions may be less accurate due to limited training data")
            
            return predictor
            
        except Exception as fallback_error:
            logger.error(f"Fallback training also failed: {fallback_error}")
            raise RuntimeError(f"Training failed: {e}")


def make_predictions(
    args: argparse.Namespace,
    logger: logging.Logger,
    predictor: NFLRushingPredictor,
    current_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Make predictions on current season players.
    
    Args:
        args: Parsed command line arguments
        logger: Logger instance
        predictor: Trained predictor instance
        current_df: Current season player data
        
    Returns:
        DataFrame with predictions
        
    Raises:
        RuntimeError: If prediction fails
    """
    logger.info("Making predictions...")
    
    try:
        # Get predictions ordered by confidence
        results = predictor.predict(current_df, sort_by='confidence')
        
        logger.info(f"Generated predictions for {len(results)} players")
        
        # Apply top-N uplift if requested
        if args.top_n > 0 and args.top_n_multiplier != 1.0:
            logger.info(
                f"Applying {args.top_n_multiplier}x uplift to top {args.top_n} players"
            )
            results = predictor.apply_top_n_uplift(
                results,
                top_n=args.top_n,
                multiplier=args.top_n_multiplier
            )
        
        # Log top predictions
        if not results.empty:
            top_player = results.iloc[0]
            logger.info(
                f"Top prediction: {top_player['player_name']} - "
                f"{top_player['predicted_rushing_yards']:.0f} yards"
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to make predictions: {e}")


def save_outputs(
    args: argparse.Namespace,
    logger: logging.Logger,
    predictor: NFLRushingPredictor,
    results: pd.DataFrame
) -> None:
    """
    Save predictions and optionally the model.
    
    Args:
        args: Parsed command line arguments
        logger: Logger instance
        predictor: Trained predictor instance
        results: Predictions DataFrame
    """
    # Save predictions to CSV
    output_path = args.output_dir / args.output_file
    
    try:
        success = export_predictions_to_csv(results, output_path, include_metadata=True)
        
        if success:
            logger.info(f"✓ Saved predictions to {output_path}")
        else:
            logger.error(f"✗ Failed to save predictions to {output_path}")
            
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
    
    # Save model if requested
    if args.save_model:
        try:
            model_path = MODELS_DIR / f"nfl_predictor_{args.model_type}.pkl"
            predictor.save_model(model_path)
            logger.info(f"✓ Saved model to {model_path}")
        except Exception as e:
            logger.error(f"✗ Failed to save model: {e}")


def main() -> int:
    """
    Main execution function.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else args.log_level
    logger = setup_logging(
        log_level=log_level,
        log_file=str(LOGGING_CONFIG['filename'])
    )
    
    logger.info("=" * 70)
    logger.info("NFL Rushing Predictor - Starting Run")
    logger.info("=" * 70)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Blend method: {args.blend_method}")
    logger.info(f"Output file: {args.output_file}")
    logger.info("")
    
    try:
        # Load data
        hist_df, current_df = load_data(args, logger)
        
        # Train model
        predictor = train_model(args, logger, hist_df)
        
        # Make predictions
        results = make_predictions(args, logger, predictor, current_df)
        
        # Save outputs
        save_outputs(args, logger, predictor, results)
        
        # Display formatted results
        print("\n")
        print(format_predictions_output(results, max_players=args.display_top_n))
        print("\n")
        
        logger.info("=" * 70)
        logger.info("NFL Rushing Predictor - Completed Successfully!")
        logger.info("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error(f"FATAL ERROR: {e}")
        logger.error(f"{'='*70}")
        logger.error("Execution failed", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())