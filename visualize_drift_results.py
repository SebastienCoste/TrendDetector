#!/usr/bin/env python3
"""
Simple visualization tool for drift test results
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse

def load_drift_results(file_path):
    """Load drift test results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_simple_drift_plot(data):
    """Create a simple drift visualization"""
    
    records = data['records']
    df = pd.DataFrame(records)
    
    # Calculate time in hours
    df['time_hours'] = (df['timestamp'] - df['timestamp'].min()) / 3600
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"Concept Drift Analysis - {data['config']['model_type'].title()} Model", fontsize=14)
    
    # Plot 1: Prediction error over time
    ax1 = axes[0]
    
    # Separate pre-drift and drift periods
    pre_drift = df[~df['is_drift_period']]
    drift_period = df[df['is_drift_period']]
    
    # Plot error
    if len(pre_drift) > 0:
        ax1.plot(pre_drift['time_hours'], pre_drift['prediction_error'], 
                'b-', alpha=0.7, linewidth=2, label='Pre-drift period')
    
    if len(drift_period) > 0:
        ax1.plot(drift_period['time_hours'], drift_period['prediction_error'], 
                'r-', alpha=0.7, linewidth=2, label='Drift period')
        
        # Mark drift introduction
        drift_start = drift_period['time_hours'].min()
        ax1.axvline(x=drift_start, color='orange', linestyle='--', 
                   linewidth=2, label='Drift introduced')
    
    # Mark feedback points
    feedback_points = df[df['feedback_provided']]
    if len(feedback_points) > 0:
        ax1.scatter(feedback_points['time_hours'], feedback_points['prediction_error'],
                   c='green', marker='o', s=40, alpha=0.8, label='Feedback provided')
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Prediction Error')
    ax1.set_title('Prediction Error Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: True vs predicted values
    ax2 = axes[1]
    
    if data['config']['model_type'] == 'classification':
        # Map categories to numbers
        trend_map = {'upward': 1, 'neutral': 0, 'downward': -1}
        df['true_numeric'] = df['true_trend'].map(trend_map)
        df['pred_numeric'] = df['predicted_trend'].map(trend_map)
        
        ax2.plot(df['time_hours'], df['true_numeric'], 'g-', 
                linewidth=2, label='True trend', alpha=0.8)
        ax2.plot(df['time_hours'], df['pred_numeric'], 'b--', 
                linewidth=1.5, label='Predicted trend', alpha=0.8)
        ax2.set_ylabel('Trend Category')
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['Downward', 'Neutral', 'Upward'])
    else:
        # Regression - plot continuous values
        ax2.plot(df['time_hours'], df['true_trend'], 'g-', 
                linewidth=2, label='True score', alpha=0.8)
        ax2.plot(df['time_hours'], df['predicted_trend'], 'b--', 
                linewidth=1.5, label='Predicted score', alpha=0.8)
        ax2.set_ylabel('Trend Score')
    
    # Mark drift point
    if len(drift_period) > 0:
        ax2.axvline(x=drift_start, color='orange', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Drift introduced')
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_title('True vs Predicted Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add text summary
    analysis = data['analysis']
    summary_text = f"""
Test Summary:
• Pre-drift error: {analysis['pre_drift_error']['mean']:.4f}
• Drift error: {analysis['drift_error']['mean']:.4f}
• Error change: {analysis.get('error_increase_percent', 'N/A'):.1f}%
• Drift detections: {analysis['drift_detected_count']}
• Total samples: {analysis['total_samples']}
"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize drift test results")
    parser.add_argument("--file", "-f", type=str, required=True, help="JSON results file to visualize")
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"Error: File {args.file} not found")
        return
    
    print(f"Loading drift results from: {args.file}")
    data = load_drift_results(args.file)
    
    print(f"Model type: {data['config']['model_type']}")
    print(f"Total samples: {len(data['records'])}")
    print(f"Drift magnitude: {data['config']['drift_magnitude']}")
    
    create_simple_drift_plot(data)

if __name__ == "__main__":
    main()