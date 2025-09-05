#!/usr/bin/env python3
"""
Quick runner for drift evaluation tests
"""

import argparse
import sys
from pathlib import Path

# Add tests to path
sys.path.insert(0, str(Path(__file__).parent / 'tests'))

from drift_evaluation import run_drift_evaluation_test, DriftTestConfig, DriftEvaluationTest

def main():
    parser = argparse.ArgumentParser(description="Run concept drift evaluation test")
    parser.add_argument("--model-type", type=str, choices=["classification", "regression"], 
                        default="classification", help="Model type to test")
    parser.add_argument("--n-initial", type=int, default=100, help="Number of initial samples")
    parser.add_argument("--n-drift", type=int, default=150, help="Number of drift samples")
    parser.add_argument("--drift-magnitude", type=float, default=0.5, help="Drift magnitude")
    parser.add_argument("--feedback-freq", type=int, default=10, help="Feedback frequency")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer samples")
    
    args = parser.parse_args()
    
    if args.quick:
        args.n_initial = 50
        args.n_drift = 75
    
    print(f"Running drift evaluation test...")
    print(f"Model type: {args.model_type}")
    print(f"Initial samples: {args.n_initial}")
    print(f"Drift samples: {args.n_drift}")
    print(f"Drift magnitude: {args.drift_magnitude}")
    print(f"Feedback frequency: {args.feedback_freq}")
    print("-" * 50)
    
    test, analysis = run_drift_evaluation_test(
        model_type=args.model_type,
        n_initial=args.n_initial,
        n_drift=args.n_drift,
        drift_magnitude=args.drift_magnitude,
        feedback_freq=args.feedback_freq
    )
    
    print("\n" + "="*50)
    print("DRIFT TEST COMPLETED")
    print("="*50)
    print("Key Results:")
    print(f"- Total samples: {analysis['total_samples']}")
    print(f"- Pre-drift error: {analysis['pre_drift_error']['mean']:.4f}")
    print(f"- Drift error: {analysis['drift_error']['mean']:.4f}")
    if 'error_increase_percent' in analysis:
        print(f"- Error increase: {analysis['error_increase_percent']:.2f}%")
    print(f"- Drift detections: {analysis['drift_detected_count']}")
    print(f"- Results saved to: {test.output_dir}")

if __name__ == "__main__":
    main()