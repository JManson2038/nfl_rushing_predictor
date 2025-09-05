"""
Utility functions for NFL Rushing Predictor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from pathlib import Path

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
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
    """Format predictions for display"""
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
    """Create feature importance visualization"""
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
    """Create scatter plot comparing predictions vs previous year performance"""
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
    """Create