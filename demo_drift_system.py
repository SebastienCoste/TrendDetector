#!/usr/bin/env python3
"""
Demonstration of the automated concept drift evaluation system
"""

import sys
from pathlib import Path
import time

# Add tests to path  
sys.path.insert(0, str(Path(__file__).parent / 'tests'))

from drift_evaluation import DriftTestConfig, DriftEvaluationTest

def demo_drift_evaluation_system():
    """Demonstrate the complete drift evaluation system"""
    
    print("🎯 AUTOMATED CONCEPT DRIFT EVALUATION SYSTEM DEMO")
    print("=" * 60)
    print()
    
    print("This system tests how our TrendDetector models handle concept drift:")
    print("• Trains a model on initial data")
    print("• Runs inference with regular feedback")  
    print("• Introduces concept drift")
    print("• Monitors adaptation and performance")
    print("• Provides detailed analysis and visualization")
    print()
    
    # Demo configuration
    print("📋 Demo Configuration:")
    config = DriftTestConfig(
        model_type="classification",
        model_name="demo_drift_model",
        n_initial_samples=60,
        n_drift_samples=80, 
        drift_magnitude=0.7,
        feedback_frequency=8,
        output_dir="./demo_drift_results",
        save_plots=True,
        save_data=True
    )
    
    print(f"   • Model Type: {config.model_type}")
    print(f"   • Initial Samples: {config.n_initial_samples}")
    print(f"   • Drift Samples: {config.n_drift_samples}")
    print(f"   • Drift Magnitude: {config.drift_magnitude}")
    print(f"   • Feedback Frequency: Every {config.feedback_frequency} samples")
    print()
    
    # Create test instance
    print("🔧 Initializing Test System...")
    test = DriftEvaluationTest(config)
    
    # Run experiment
    print("🚀 Running Drift Evaluation Experiment...")
    print("-" * 40)
    
    start_time = time.time()
    test.run_complete_experiment()
    end_time = time.time()
    
    print(f"⏱️  Experiment completed in {end_time - start_time:.2f} seconds")
    print()
    
    # Analyze results
    print("📊 Analyzing Results...")
    analysis = test.analyze_results()
    
    # Create visualizations (will open plot window)
    print("📈 Creating Visualizations...")
    test.create_visualizations(analysis)
    
    # Save results
    print("💾 Saving Results...")
    test.save_results(analysis)
    
    # Print detailed summary
    print()
    print("🎉 DRIFT EVALUATION COMPLETED!")
    print("=" * 60)
    print()
    
    print("📈 KEY PERFORMANCE METRICS:")
    print(f"   • Pre-drift Error:    {analysis['pre_drift_error']['mean']:.4f} ± {analysis['pre_drift_error']['std']:.4f}")
    print(f"   • Drift Period Error: {analysis['drift_error']['mean']:.4f} ± {analysis['drift_error']['std']:.4f}")
    
    if 'error_increase_percent' in analysis:
        change_direction = "↗️ Increase" if analysis['error_increase_percent'] > 0 else "↘️ Decrease"
        print(f"   • Error Change:      {abs(analysis['error_increase_percent']):.2f}% {change_direction}")
    
    print(f"   • Drift Detections:   {analysis['drift_detected_count']}")
    print(f"   • Feedback Events:    {analysis['feedback_provided_count']}")
    print(f"   • Total Samples:      {analysis['total_samples']}")
    print()
    
    print("💡 SYSTEM CAPABILITIES DEMONSTRATED:")
    print("   ✅ Automated experiment orchestration")
    print("   ✅ Realistic concept drift simulation")  
    print("   ✅ Real-time model adaptation monitoring")
    print("   ✅ Comprehensive performance analysis")
    print("   ✅ Detailed visualization and reporting")
    print("   ✅ Configurable test parameters")
    print("   ✅ Reproducible experiments")
    print("   ✅ Both classification and regression support")
    print()
    
    print(f"📁 Results saved to: {test.output_dir}")
    print()
    
    # Interpretation
    print("🔍 RESULT INTERPRETATION:")
    
    if analysis.get('error_increase_percent', 0) > 10:
        print("   📈 Significant error increase detected during drift period")
        print("   🔧 Model may benefit from more frequent adaptation")
    elif analysis.get('error_increase_percent', 0) < -10:
        print("   📉 Model performance improved during drift period") 
        print("   ✨ Continuous learning is working effectively")
    else:
        print("   📊 Model maintained stable performance during drift")
        print("   🎯 Good adaptation capabilities demonstrated")
    
    if analysis['drift_detected_count'] > 0:
        print(f"   🚨 Drift detection system triggered {analysis['drift_detected_count']} times")
        print("   ⚡ Automatic adaptation mechanisms activated")
    else:
        print("   🤔 No explicit drift detections (may indicate gradual adaptation)")
    
    print()
    print("🎪 Demo completed! Check the generated plots for detailed analysis.")
    
    return test, analysis

def demo_regression_vs_classification():
    """Quick comparison between regression and classification drift handling"""
    
    print("\n" + "="*60)
    print("🔬 REGRESSION vs CLASSIFICATION DRIFT COMPARISON")
    print("="*60)
    
    results = {}
    
    for model_type in ['classification', 'regression']:
        print(f"\nTesting {model_type.upper()} model...")
        
        config = DriftTestConfig(
            model_type=model_type,
            model_name=f"comparison_{model_type}",
            n_initial_samples=40,
            n_drift_samples=60,
            drift_magnitude=0.5,
            feedback_frequency=10,
            output_dir=f"./comparison_results_{model_type}"
        )
        
        test = DriftEvaluationTest(config)
        test.run_complete_experiment()
        analysis = test.analyze_results()
        
        results[model_type] = analysis
        
        print(f"✓ {model_type.title()} completed:")
        print(f"  Pre-drift error: {analysis['pre_drift_error']['mean']:.4f}")
        print(f"  Drift error: {analysis['drift_error']['mean']:.4f}")
        print(f"  Error change: {analysis.get('error_increase_percent', 0):.2f}%")
    
    # Compare results
    print("\n📊 COMPARISON SUMMARY:")
    print("-" * 30)
    
    for model_type in ['classification', 'regression']:
        analysis = results[model_type]
        robustness = "High" if abs(analysis.get('error_increase_percent', 0)) < 10 else "Medium"
        print(f"{model_type.title():>14}: {robustness} robustness ({analysis.get('error_increase_percent', 0):+.1f}% change)")
    
    return results

if __name__ == "__main__":
    try:
        # Main demo
        test, analysis = demo_drift_evaluation_system()
        
        # Optional: Run comparison
        print("\n" + "?"*60)
        response = input("Run regression vs classification comparison? (y/n): ")
        if response.lower() in ['y', 'yes']:
            demo_regression_vs_classification()
        
        print("\n🎉 All demonstrations completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()