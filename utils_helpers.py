"""Utility functions for NFL Rushing Predictor with safety enhancements."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
import warnings


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration with rotation.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Validate log level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    log_level = log_level.upper()
    
    if log_level not in valid_levels:
        print(f"Invalid log level '{log_level}'. Using INFO.")
        log_level = 'INFO'
    
    # Create handlers
    handlers = [logging.StreamHandler()]
    
    if log_file:
        try:
            # Use rotating file handler to prevent huge log files
            from logging.handlers import RotatingFileHandler
            
            # Create log directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            )
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file: {e}")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {log_level} level")
    
    return logger


def format_predictions_output(results: pd.DataFrame, max_players: int = 10) -> str:
    """
    Format predictions for display with safety checks.
    
    Args:
        results: Predictions DataFrame
        max_players: Maximum number of players to display
        
    Returns:
        Formatted string output
    """
    if results is None or results.empty:
        return "No predictions available."
    
    output = []
    output.append("🏆 2025 NFL RUSHING LEADER PREDICTIONS")
    output.append("=" * 70)
    output.append("")
    
    # Limit to max_players
    display_results = results.head(max_players)
    
    for idx, row in display_results.iterrows():
        try:
            rank = idx + 1
            player = row.get('player_name', 'Unknown')
            team = row.get('team', 'N/A')
            pred_yards = int(row.get('predicted_rushing_yards', 0))
            age = int(row.get('age', 0))
            
            # Add trend indicator if available
            trend = ""
            if 'predicted_change' in row:
                change = row['predicted_change']
                if pd.notna(change):
                    if change > 100:
                        trend = " ↗️ Trending Up"
                    elif change < -100:
                        trend = " ↘️ Trending Down"
                    else:
                        trend = " ➡️ Stable"
            
            # Main prediction line
            output.append(f"{rank:2d}. {player:20s} ({team}) - {pred_yards:,} yards{trend}")
            
            # Additional details
            prev_yards = int(row.get('previous_year_yards', 0))
            output.append(f"    Age: {age} | Previous: {prev_yards:,} yards")
            
            # Confidence and risk if available
            details = []
            if 'prediction_confidence' in row and pd.notna(row['prediction_confidence']):
                confidence_pct = row['prediction_confidence'] * 100
                details.append(f"Confidence: {confidence_pct:.0f}%")
            
            if 'injury_risk' in row and pd.notna(row['injury_risk']):
                risk_value = row['injury_risk']
                if risk_value > 0.7:
                    risk_level = "High"
                elif risk_value > 0.3:
                    risk_level = "Medium"
                else:
                    risk_level = "Low ✓"
                details.append(f"Injury Risk: {risk_level}")
            
            if 'predicted_change_pct' in row and pd.notna(row['predicted_change_pct']):
                change_pct = row['predicted_change_pct']
                details.append(f"Change: {change_pct:+.1f}%")
            
            if details:
                output.append(f"    {' | '.join(details)}")
            
            output.append("")
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Error formatting row {idx}: {e}")
            continue
    
    # Add summary statistics
    if len(results) > max_players:
        output.append(f"... and {len(results) - max_players} more players")
        output.append("")
    
    # Add metadata
    output.append("=" * 70)
    output.append(f"Total predictions: {len(results)}")
    
    if 'predicted_rushing_yards' in results.columns:
        avg_prediction = results['predicted_rushing_yards'].mean()
        output.append(f"Average predicted yards: {avg_prediction:.0f}")
    
    return "\n".join(output)


def create_feature_importance_plot(
    importance_df: pd.DataFrame, 
    top_n: int = 15, 
    save_path: Optional[str] = None
) -> None:
    """
    Create feature importance visualization with safety checks.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        save_path: Optional path to save the plot
    """
    if importance_df is None or importance_df.empty:
        logging.getLogger(__name__).warning("No feature importance data available")
        return
    
    if 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
        logging.getLogger(__name__).error("importance_df must have 'feature' and 'importance' columns")
        return
    
    try:
        plt.figure(figsize=(12, 8))
        
        top_features = importance_df.head(top_n)
        
        # Create bar plot
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features for NFL Rushing Prediction', 
                 fontsize=16, pad=20, fontweight='bold')
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(top_features.iterrows()):
            plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}', 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.getLogger(__name__).info(f"Saved plot to {save_path}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to save plot: {e}")
        
        plt.show()
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error creating feature importance plot: {e}")
    finally:
        plt.close()


def create_predictions_comparison_plot(
    results: pd.DataFrame, 
    save_path: Optional[str] = None
) -> None:
    """
    Create scatter plot comparing predictions vs previous year performance.
    
    Args:
        results: Predictions DataFrame
        save_path: Optional path to save the plot
    """
    if results is None or results.empty:
        logging.getLogger(__name__).warning("No results available for comparison plot")
        return
    
    if 'previous_year_yards' not in results.columns:
        logging.getLogger(__name__).warning("Previous year data not available for comparison plot")
        return
    
    if 'predicted_rushing_yards' not in results.columns:
        logging.getLogger(__name__).error("Predicted yards not available")
        return
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Filter out rows with missing data
        plot_data = results.dropna(subset=['previous_year_yards', 'predicted_rushing_yards'])
        
        if plot_data.empty:
            logging.getLogger(__name__).warning("No valid data for comparison plot")
            return
        
        # Create scatter plot with age coloring if available
        if 'age' in plot_data.columns:
            scatter = plt.scatter(
                plot_data['previous_year_yards'], 
                plot_data['predicted_rushing_yards'], 
                alpha=0.7, 
                s=100, 
                c=plot_data['age'], 
                cmap='viridis'
            )
            plt.colorbar(scatter, label='Player Age')
        else:
            plt.scatter(
                plot_data['previous_year_yards'], 
                plot_data['predicted_rushing_yards'], 
                alpha=0.7, 
                s=100
            )
        
        # Add diagonal line (y=x) for reference
        min_val = min(plot_data['previous_year_yards'].min(), 
                     plot_data['predicted_rushing_yards'].min())
        max_val = max(plot_data['previous_year_yards'].max(), 
                     plot_data['predicted_rushing_yards'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, 
                label='No Change Line')
        
        # Customize plot
        plt.xlabel('2024 Rushing Yards', fontsize=12)
        plt.ylabel('Predicted 2025 Rushing Yards', fontsize=12)
        plt.title('2025 Rushing Predictions vs 2024 Performance', 
                 fontsize=16, pad=20, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Annotate top players
        top_players = plot_data.nlargest(5, 'predicted_rushing_yards')
        for idx, row in top_players.iterrows():
            if 'player_name' in row:
                plt.annotate(
                    row['player_name'], 
                    (row['previous_year_yards'], row['predicted_rushing_yards']),
                    xytext=(5, 5), 
                    textcoords='offset points', 
                    fontsize=9,
                    alpha=0.8
                )
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.getLogger(__name__).info(f"Saved plot to {save_path}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to save plot: {e}")
        
        plt.show()
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error creating comparison plot: {e}")
    finally:
        plt.close()


def create_age_distribution_plot(
    results: pd.DataFrame, 
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of player ages with safety checks.
    
    Args:
        results: Predictions DataFrame
        save_path: Optional path to save the plot
    """
    if results is None or results.empty:
        logging.getLogger(__name__).warning("No results available for age distribution plot")
        return
    
    if 'age' not in results.columns:
        logging.getLogger(__name__).warning("Age data not available")
        return
    
    try:
        plt.figure(figsize=(10, 6))
        
        # Filter valid ages
        age_data = results['age'].dropna()
        age_data = age_data[(age_data >= 18) & (age_data <= 40)]
        
        if age_data.empty:
            logging.getLogger(__name__).warning("No valid age data")
            return
        
        sns.histplot(age_data, bins=15, kde=True, color='skyblue')
        plt.title('Age Distribution of Predicted Rushing Leaders', 
                 fontsize=16, pad=20, fontweight='bold')
        plt.xlabel('Age', fontsize=12)
        plt.ylabel('Number of Players', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add mean and median lines
        mean_age = age_data.mean()
        median_age = age_data.median()
        plt.axvline(mean_age, color='red', linestyle='--', 
                   label=f'Mean: {mean_age:.1f}', alpha=0.7)
        plt.axvline(median_age, color='green', linestyle='--', 
                   label=f'Median: {median_age:.1f}', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.getLogger(__name__).info(f"Saved plot to {save_path}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to save plot: {e}")
        
        plt.show()
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error creating age distribution plot: {e}")
    finally:
        plt.close()


def update_predictions_with_weekly_stats(
    results: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    weeks: List[int] = [1, 2, 3],
    alpha: float = 0.6,
    season_games: int = 17
) -> pd.DataFrame:
    """
    Update/adjust model predictions using actual weekly rushing stats with safety checks.

    Args:
        results: Predictions dataframe. Must contain 'player_name' and 'predicted_rushing_yards'.
        weekly_stats: DataFrame with columns ['player_name', 'week', 'rushing_yards'].
        weeks: List of week numbers to apply (default [1,2,3]).
        alpha: Blending weight for the original model prediction (0..1).
        season_games: Number of games in the season (default 17).

    Returns:
        DataFrame with updated predictions and weekly statistics.
        
    Raises:
        ValueError: If required columns are missing
    """
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if results is None or results.empty:
        logger.warning("Empty results DataFrame provided")
        return results
    
    if weekly_stats is None or weekly_stats.empty:
        logger.warning("Empty weekly_stats DataFrame provided")
        return results
    
    # Validate required columns
    if 'player_name' not in results.columns:
        raise ValueError("results must include 'player_name' column")
    if 'predicted_rushing_yards' not in results.columns:
        raise ValueError("results must include 'predicted_rushing_yards' column")
    if not {'player_name', 'week', 'rushing_yards'}.issubset(weekly_stats.columns):
        raise ValueError("weekly_stats must include 'player_name', 'week', and 'rushing_yards' columns")
    
    # Validate parameters
    if not 0 <= alpha <= 1:
        logger.warning(f"alpha should be between 0 and 1. Got {alpha}. Clipping to valid range.")
        alpha = np.clip(alpha, 0, 1)
    
    if season_games <= 0 or season_games > 20:
        logger.warning(f"Invalid season_games: {season_games}. Using default 17.")
        season_games = 17
    
    try:
        # Filter weekly stats for selected weeks
        ws = weekly_stats[weekly_stats['week'].isin(weeks)].copy()
        
        if ws.empty:
            logger.warning(f"No weekly stats found for weeks {weeks}")
            return results
        
        # Pivot to get week columns
        pivot = ws.pivot_table(
            index='player_name', 
            columns='week', 
            values='rushing_yards', 
            aggfunc='sum'
        )
        
        # Rename columns to weekX_yards
        pivot = pivot.rename(columns=lambda c: f'week{int(c)}_yards')
        
        # Merge into results
        merged = results.merge(pivot, how='left', left_on='player_name', right_index=True)
        
        # Ensure week columns exist and fill NaN with 0
        for w in weeks:
            col = f'week{w}_yards'
            if col not in merged.columns:
                merged[col] = 0
            merged[col] = merged[col].fillna(0).astype(float)
        
        # Compute actual total for the chosen weeks
        week_cols = [f'week{w}_yards' for w in weeks]
        merged['actual_first_n_total'] = merged[week_cols].sum(axis=1)
        
        # Project full season from pace observed in the selected weeks
        n_weeks = len(weeks)
        if n_weeks > 0:
            merged['projected_from_weeks'] = (
                merged['actual_first_n_total'] * (season_games / n_weeks)
            ).round(1)
        else:
            merged['projected_from_weeks'] = 0
        
        # Original model prediction (ensure numeric)
        merged['model_predicted'] = pd.to_numeric(
            merged['predicted_rushing_yards'], 
            errors='coerce'
        ).fillna(0).astype(float)
        
        # Blended adjusted prediction
        merged['adjusted_predicted_rushing_yards'] = (
            (alpha * merged['model_predicted']) + 
            ((1 - alpha) * merged['projected_from_weeks'])
        ).round().astype(int)
        
        # Clip to reasonable range
        merged['adjusted_predicted_rushing_yards'] = merged['adjusted_predicted_rushing_yards'].clip(0, 3000)
        
        # Calculate differences and percent change
        merged['adjustment_diff'] = (
            merged['adjusted_predicted_rushing_yards'] - merged['model_predicted']
        )
        
        # Avoid divide-by-zero for percent change
        merged['adjustment_pct'] = merged.apply(
            lambda r: ((r['adjusted_predicted_rushing_yards'] / r['model_predicted'] - 1) * 100)
            if r['model_predicted'] > 0 else np.nan,
            axis=1
        ).round(1)
        
        # Add metadata
        merged['weeks_used'] = ",".join(str(w) for w in weeks)
        merged['alpha_used'] = alpha
        
        # Sort by adjusted predictions
        merged = merged.sort_values('adjusted_predicted_rushing_yards', ascending=False)
        merged = merged.reset_index(drop=True)
        
        logger.info(f"Updated predictions with weekly stats from weeks {weeks}")
        
        return merged
        
    except Exception as e:
        logger.error(f"Error updating predictions with weekly stats: {e}", exc_info=True)
        return results


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame"
) -> Tuple[bool, List[str]]:
    """
    Validate that a DataFrame has required columns and is not empty.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name of the DataFrame for error messages
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if df is None:
        errors.append(f"{name} is None")
        return False, errors
    
    if df.empty:
        errors.append(f"{name} is empty")
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"{name} missing required columns: {missing_cols}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def safe_divide(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    default: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely divide two values, handling division by zero.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Default value to return when denominator is zero
        
    Returns:
        Result of division or default value
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if isinstance(denominator, (pd.Series, np.ndarray)):
            result = np.divide(
                numerator, 
                denominator, 
                out=np.full_like(numerator, default, dtype=float),
                where=denominator != 0
            )
        else:
            result = numerator / denominator if denominator != 0 else default
        
        return result


def export_predictions_to_csv(
    results: pd.DataFrame,
    filepath: Path,
    include_metadata: bool = True
) -> bool:
    """
    Export predictions to CSV with safety checks.
    
    Args:
        results: Predictions DataFrame
        filepath: Path to save CSV
        include_metadata: Whether to include metadata columns
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if results is None or results.empty:
        logger.error("Cannot export empty results")
        return False
    
    try:
        # Create directory if it doesn't exist
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Select columns to export
        if include_metadata:
            export_df = results.copy()
        else:
            # Export only essential columns
            essential_cols = [
                'player_name', 'team', 'age', 
                'predicted_rushing_yards', 'previous_year_yards'
            ]
            available_cols = [col for col in essential_cols if col in results.columns]
            export_df = results[available_cols].copy()
        
        # Save to CSV
        export_df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(export_df)} predictions to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export predictions: {e}")
        return False
