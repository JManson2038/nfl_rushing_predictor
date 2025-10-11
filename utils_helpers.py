#Utility functions for NFL Rushing Predictor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from pathlib import Path

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    #Setup logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def format_predictions_output(results: pd.DataFrame) -> str:
    #Format predictions for display
    output = []
    output.append("🏆 2025 NFL RUSHING LEADER PREDICTIONS")
    output.append("=" * 70)
    
    for idx, row in results.iterrows():
        rank = idx + 1
        player = row['player_name']
        team = row['team']
        pred_yards = int(row['predicted_rushing_yards'])
        age = int(row['age'])
        
        # Add trend indicator if available
        trend = ""
        if 'predicted_change' in row:
            change = row['predicted_change']
            if change > 100:
                trend = " ↗️"
            elif change < -100:
                trend = " ↘️"
            else:
                trend = " ➡️"
        
        output.append(f"{rank:2d}. {player:20s} ({team}) - {pred_yards:,} yards{trend}")
        output.append(f"    Age: {age} | Previous: {int(row.get('previous_year_yards', 0)):,} yards")
        
        # Add confidence and risk if available
        details = []
        if 'prediction_confidence' in row:
            details.append(f"Confidence: {row['prediction_confidence']:.2f}")
        if 'injury_risk' in row:
            risk_level = "High" if row['injury_risk'] > 0.7 else "Medium" if row['injury_risk'] > 0.3 else "Low"
            details.append(f"Injury Risk: {risk_level}")
        
        if details:
            output.append(f"    {' | '.join(details)}")
        
        output.append("")
    
    return "\n".join(output)

def create_feature_importance_plot(importance_df: pd.DataFrame, top_n: int = 15, save_path: Optional[str] = None):
    #Create feature importance visualization
    plt.figure(figsize=(12, 8))
    
    top_features = importance_df.head(top_n)
    
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title(f'Top {top_n} Most Important Features for NFL Rushing Prediction', fontsize=16, pad=20)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_predictions_comparison_plot(results: pd.DataFrame, save_path: Optional[str] = None):
    #Create scatter plot comparing predictions vs previous year performance
    if 'previous_year_yards' not in results.columns:
        print("Previous year data not available for comparison plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(results['previous_year_yards'], results['predicted_rushing_yards'], 
               alpha=0.7, s=100, c=results['age'], cmap='viridis')
    
    # Add diagonal line (y=x) for reference
    min_val = min(results['previous_year_yards'].min(), results['predicted_rushing_yards'].min())
    max_val = max(results['previous_year_yards'].max(), results['predicted_rushing_yards'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='No Change Line')
    
    # Customize plot
    plt.xlabel('2024 Rushing Yards', fontsize=12)
    plt.ylabel('Predicted 2025 Rushing Yards', fontsize=12)
    plt.title('2025 Rushing Predictions vs 2024 Performance', fontsize=16, pad=20)
    plt.colorbar(label='Player Age')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Annotate top players
    for idx, row in results.head(5).iterrows():
        plt.annotate(row['player_name'], 
                    (row['previous_year_yards'], row['predicted_rushing_yards']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_age_distribution_plot(results: pd.DataFrame, save_path: Optional[str] = None):
    #Plot distribution of player ages
    plt.figure(figsize=(10, 6))
    
    sns.histplot(results['age'], bins=15, kde=True, color='skyblue')
    plt.title('Age Distribution of Predicted Rushing Leaders', fontsize=16, pad=20)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Number of Players', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# New: update predictions with weekly stats (e.g. weeks 1-3 of 2025)
def update_predictions_with_weekly_stats(
    results: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    weeks: List[int] = [1, 2, 3],
    alpha: float = 0.6,
    season_games: int = 17
) -> pd.DataFrame:
    """
    Update/adjust model predictions using actual weekly rushing stats.

    Parameters
    - results: predictions dataframe produced by the model. Must contain 'player_name' and 'predicted_rushing_yards'.
    - weekly_stats: DataFrame with columns ['player_name', 'week', 'rushing_yards'] (can include other cols).
    - weeks: list of week numbers to apply (default [1,2,3]).
    - alpha: blending weight for the original model prediction (0..1). adjusted = alpha*model + (1-alpha)*pace_projection
    - season_games: number of games in the season (default 17)

    Returns:
    - DataFrame with added columns per-week, aggregate of the chosen weeks, projected full-season from pace,
      and an adjusted model prediction.
    """
    # Ensure inputs
    if 'player_name' not in results.columns:
        raise ValueError("results must include 'player_name' column")
    if 'predicted_rushing_yards' not in results.columns:
        raise ValueError("results must include 'predicted_rushing_yards' column")
    if not {'player_name', 'week', 'rushing_yards'}.issubset(weekly_stats.columns):
        raise ValueError("weekly_stats must include 'player_name', 'week', and 'rushing_yards' columns")
    
    # Aggregate weekly stats for selected weeks
    ws = weekly_stats[weekly_stats['week'].isin(weeks)].copy()
    # Pivot so we get week1_yards, week2_yards, ...
    pivot = ws.pivot_table(index='player_name', columns='week', values='rushing_yards', aggfunc='sum')
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
    merged['projected_from_weeks'] = (merged['actual_first_n_total'] * (season_games / n_weeks)).round(1)
    
    # Original model prediction (ensure numeric)
    merged['model_predicted'] = merged['predicted_rushing_yards'].fillna(0).astype(float)
    
    # Blended adjusted prediction
    merged['adjusted_predicted_rushing_yards'] = (
        (alpha * merged['model_predicted']) + ((1 - alpha) * merged['projected_from_weeks'])
    ).round().astype(int)
    
    # Differences and percent change
    merged['adjustment_diff'] = merged['adjusted_predicted_rushing_yards'] - merged['model_predicted']
    # avoid divide-by-zero
    merged['adjustment_pct'] = merged.apply(
        lambda r: ((r['adjusted_predicted_rushing_yards'] / r['model_predicted'] - 1) * 100)
        if r['model_predicted'] > 0 else np.nan,
        axis=1
    ).round(1)
    
    # Helpful summary columns
    merged['weeks_used'] = ",".join(str(w) for w in weeks)
    merged = merged.sort_values('adjusted_predicted_rushing_yards', ascending=False)
    
    return merged
