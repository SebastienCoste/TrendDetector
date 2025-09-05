#!/usr/bin/env python3
"""
Comprehensive drift evaluation testing with multiple scenarios
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Add tests to path
sys.path.insert(0, str(Path(__file__).parent / 'tests'))

from drift_evaluation import DriftTestConfig, DriftEvaluationTest

def run_batch_drift_tests():
    """Run multiple drift test scenarios"""
    
    results = []
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Low Drift Classification',
            'model_type': 'classification',
            'drift_magnitude': 0.3,
            'n_initial': 80,
            'n_drift': 100
        },
        {
            'name': 'High Drift Classification',
            'model_type': 'classification', 
            'drift_magnitude': 0.8,
            'n_initial': 80,
            'n_drift': 100
        },
        {
            'name': 'Low Drift Regression',
            'model_type': 'regression',
            'drift_magnitude': 0.3,
            'n_initial': 80,
            'n_drift': 100
        },
        {
            'name': 'High Drift Regression',
            'model_type': 'regression',
            'drift_magnitude': 0.8,
            'n_initial': 80,
            'n_drift': 100
        }
    ]
    
    print("Running comprehensive drift evaluation tests...")
    print("="*60)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nTest {i}/4: {scenario['name']}")
        print("-" * 40)
        
        config = DriftTestConfig(
            model_type=scenario['model_type'],
            model_name=f"drift_test_{scenario['model_type']}_{i}",
            n_initial_samples=scenario['n_initial'],
            n_drift_samples=scenario['n_drift'],
            drift_magnitude=scenario['drift_magnitude'],
            feedback_frequency=15,
            output_dir=f"./batch_drift_results/scenario_{i}_{scenario['name'].replace(' ', '_').lower()}"
        )
        
        try:
            test = DriftEvaluationTest(config)
            test.run_complete_experiment()
            analysis = test.analyze_results()
            test.save_results(analysis)
            
            # Store results for comparison
            result = {
                'scenario': scenario['name'],
                'model_type': scenario['model_type'],
                'drift_magnitude': scenario['drift_magnitude'],
                'pre_drift_error': analysis['pre_drift_error']['mean'],
                'drift_error': analysis['drift_error']['mean'],
                'error_increase_percent': analysis.get('error_increase_percent', 0),
                'drift_detected_count': analysis['drift_detected_count'],
                'total_samples': analysis['total_samples']
            }
            results.append(result)
            
            print(f"✓ Pre-drift error: {result['pre_drift_error']:.4f}")
            print(f"✓ Drift error: {result['drift_error']:.4f}")
            print(f"✓ Error change: {result['error_increase_percent']:.2f}%")
            print(f"✓ Drift detections: {result['drift_detected_count']}")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            continue
    
    # Create comparison plots
    create_comparison_plots(results)
    
    # Save batch results
    save_batch_results(results)
    
    return results

def create_comparison_plots(results):
    """Create comparison plots for all test scenarios"""
    
    if not results:
        print("No results to plot")
        return
    
    # Prepare data
    df = pd.DataFrame(results)
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Drift Evaluation Results', fontsize=16)
    
    # Plot 1: Error by drift magnitude
    ax1 = axes[0, 0]
    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type]
        ax1.scatter(subset['drift_magnitude'], subset['drift_error'], 
                   label=f'{model_type.title()}', s=100, alpha=0.7)
    
    ax1.set_xlabel('Drift Magnitude')
    ax1.set_ylabel('Drift Period Error')
    ax1.set_title('Error vs Drift Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error increase by scenario
    ax2 = axes[0, 1]
    scenarios = df['scenario'].tolist()
    error_increases = df['error_increase_percent'].tolist()
    colors = ['blue' if 'Classification' in s else 'red' for s in scenarios]
    
    bars = ax2.bar(range(len(scenarios)), error_increases, color=colors, alpha=0.7)
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Error Increase (%)')
    ax2.set_title('Error Increase by Scenario')
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels([s.replace(' ', '\n') for s in scenarios], rotation=0, fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, error_increases):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    # Plot 3: Pre-drift vs Drift error
    ax3 = axes[1, 0]
    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type]
        ax3.scatter(subset['pre_drift_error'], subset['drift_error'],
                   label=f'{model_type.title()}', s=100, alpha=0.7)
    
    # Add diagonal line
    max_error = max(df['pre_drift_error'].max(), df['drift_error'].max())
    ax3.plot([0, max_error], [0, max_error], 'k--', alpha=0.5, label='No change line')
    
    ax3.set_xlabel('Pre-drift Error')
    ax3.set_ylabel('Drift Period Error') 
    ax3.set_title('Pre-drift vs Drift Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Drift detection effectiveness
    ax4 = axes[1, 1]
    drift_counts = df['drift_detected_count'].tolist()
    
    bars = ax4.bar(range(len(scenarios)), drift_counts, color=colors, alpha=0.7)
    ax4.set_xlabel('Scenario')
    ax4.set_ylabel('Drift Detection Count')
    ax4.set_title('Drift Detection Effectiveness')
    ax4.set_xticks(range(len(scenarios)))
    ax4.set_xticklabels([s.replace(' ', '\n') for s in scenarios], rotation=0, fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, drift_counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(int(value)), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("./batch_drift_results")
    output_dir.mkdir(exist_ok=True)
    
    plot_file = output_dir / f"comprehensive_drift_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_file}")
    
    # Show plot
    plt.show()

def save_batch_results(results):
    """Save batch test results"""
    output_dir = Path("./batch_drift_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON summary
    summary = {
        'test_timestamp': datetime.now().isoformat(),
        'total_scenarios': len(results),
        'results': results,
        'summary_stats': {
            'avg_error_increase': np.mean([r['error_increase_percent'] for r in results]),
            'max_error_increase': max([r['error_increase_percent'] for r in results]),
            'min_error_increase': min([r['error_increase_percent'] for r in results]),
            'total_drift_detections': sum([r['drift_detected_count'] for r in results])
        }
    }
    
    summary_file = output_dir / f"batch_drift_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Batch results saved to: {summary_file}")
    
    # Create results table
    df = pd.DataFrame(results)
    csv_file = output_dir / f"batch_drift_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"Results CSV saved to: {csv_file}")
    
    return summary

def print_final_summary(results):
    """Print final summary of all tests"""
    print("\n" + "="*80)
    print("COMPREHENSIVE DRIFT EVALUATION SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    print(f"Total scenarios tested: {len(results)}")
    print(f"Model types: {', '.join(df['model_type'].unique())}")
    print()
    
    # Summary by model type
    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type]
        print(f"{model_type.upper()} MODEL RESULTS:")
        print(f"  Average error increase: {subset['error_increase_percent'].mean():.2f}%")
        print(f"  Max error increase: {subset['error_increase_percent'].max():.2f}%")
        print(f"  Total drift detections: {subset['drift_detected_count'].sum()}")
        print(f"  Average drift error: {subset['drift_error'].mean():.4f}")
        print()
    
    # Best and worst performers
    best_scenario = df.loc[df['error_increase_percent'].idxmin()]
    worst_scenario = df.loc[df['error_increase_percent'].idxmax()]
    
    print("BEST PERFORMING SCENARIO:")
    print(f"  {best_scenario['scenario']}")
    print(f"  Error change: {best_scenario['error_increase_percent']:.2f}%")
    print()
    
    print("WORST PERFORMING SCENARIO:")
    print(f"  {worst_scenario['scenario']}")
    print(f"  Error change: {worst_scenario['error_increase_percent']:.2f}%")
    print()
    
    # Drift detection analysis
    total_detections = df['drift_detected_count'].sum()
    print(f"DRIFT DETECTION SUMMARY:")
    print(f"  Total detections across all tests: {total_detections}")
    print(f"  Average detections per test: {total_detections / len(results):.1f}")
    
    print("="*80)

if __name__ == "__main__":
    try:
        results = run_batch_drift_tests()
        print_final_summary(results)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()