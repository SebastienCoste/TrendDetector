#!/usr/bin/env python3
"""
Automated Concept Drift Evaluation Test for TrendDetector Dual-Model System

This module implements comprehensive testing for concept drift detection and adaptation
in both classification and regression models with visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.config import AppConfig
from src.core.model_manager import initialize_model_manager
from src.core.gpu_utils import initialize_gpu
from src.ml.model_interface import TrendModelInterface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DriftTestConfig:
    """Configuration for drift evaluation test"""
    model_type: str = "classification"  # or "regression"
    model_name: str = "drift_test_model"
    n_initial_samples: int = 200
    n_drift_samples: int = 300
    feedback_frequency: int = 10
    embedding_dim: int = 512
    drift_start_point: int = 200  # When to introduce drift
    drift_magnitude: float = 0.5  # How much to shift the data generation
    output_dir: str = "./drift_test_results"
    save_plots: bool = True
    save_data: bool = True

@dataclass
class SampleRecord:
    """Record for a single sample in the drift test"""
    timestamp: float
    sample_index: int
    embedding_vector: List[float]
    true_trend: Any  # str for classification, float for regression
    predicted_trend: Any
    confidence: float
    prediction_error: float
    is_drift_period: bool
    feedback_provided: bool
    drift_detected: bool = False

class ConceptDriftSimulator:
    """Simulates concept drift by modifying data generation parameters"""
    
    def __init__(self, embedding_dim: int = 512, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.seed = seed
        np.random.seed(seed)
        
        # Base parameters for trend calculation
        self.base_weights = {
            'vector_sum_threshold': 5.0,
            'magnitude_threshold': 25.0,
            'variance_threshold': 1.5,
            'time_decay_factor': 7 * 24 * 3600,  # 7 days
            'peak_hour_boost': 0.15,
            'weekend_boost': 0.1,
            'noise_std': 0.1
        }
        
        # Current weights (will be modified during drift)
        self.current_weights = self.base_weights.copy()
        
    def reset_to_base(self):
        """Reset to original parameters"""
        self.current_weights = self.base_weights.copy()
        logger.info("Data generation reset to base parameters")
    
    def introduce_drift(self, drift_magnitude: float = 0.5):
        """Introduce concept drift by modifying generation parameters"""
        logger.info(f"Introducing concept drift with magnitude {drift_magnitude}")
        
        # Modify key parameters to simulate drift
        self.current_weights.update({
            'vector_sum_threshold': self.base_weights['vector_sum_threshold'] * (1 + drift_magnitude),
            'magnitude_threshold': self.base_weights['magnitude_threshold'] * (1 - drift_magnitude * 0.3),
            'variance_threshold': self.base_weights['variance_threshold'] * (1 + drift_magnitude * 0.2),
            'peak_hour_boost': self.base_weights['peak_hour_boost'] * (1 - drift_magnitude * 0.5),
            'weekend_boost': -self.base_weights['weekend_boost'],  # Reverse weekend effect
            'noise_std': self.base_weights['noise_std'] * (1 + drift_magnitude)
        })
        
        logger.info(f"New parameters: {self.current_weights}")
    
    def generate_embedding_vector(self, content_type: str = "random") -> np.ndarray:
        """Generate synthetic embedding vector"""
        if content_type == "random":
            return np.random.randn(self.embedding_dim)
        elif content_type == "viral":
            vector = np.random.randn(self.embedding_dim)
            viral_dims = np.random.choice(self.embedding_dim, size=self.embedding_dim//10, replace=False)
            vector[viral_dims] *= 2.0
            return vector
        elif content_type == "trending":
            vector = np.random.randn(self.embedding_dim)
            trending_dims = np.random.choice(self.embedding_dim, size=self.embedding_dim//20, replace=False)
            vector[trending_dims] *= 1.5
            return vector
        else:
            return np.random.randn(self.embedding_dim)
    
    def calculate_true_trend_score(self, vector: np.ndarray, timestamp: float, base_time: float) -> float:
        """Calculate true trend score using current (possibly drifted) parameters"""
        
        # Vector-based features
        vector_sum = np.sum(vector)
        vector_magnitude = np.linalg.norm(vector)
        vector_variance = np.var(vector)
        
        # Time-based features
        time_since_base = timestamp - base_time
        hour_of_day = (timestamp % (24 * 3600)) / 3600
        day_of_week = int(timestamp // (24 * 3600)) % 7
        
        # Calculate trend score using current weights
        trend_score = 0.0
        
        # Vector influence (modified by drift)
        if vector_sum > self.current_weights['vector_sum_threshold']:
            trend_score += 0.2
        elif vector_sum < -self.current_weights['vector_sum_threshold']:
            trend_score -= 0.2
            
        if vector_magnitude > self.current_weights['magnitude_threshold']:
            trend_score += 0.1
        elif vector_magnitude < (self.current_weights['magnitude_threshold'] * 0.6):
            trend_score -= 0.1
            
        if vector_variance > self.current_weights['variance_threshold']:
            trend_score += 0.1
        elif vector_variance < (self.current_weights['variance_threshold'] * 0.3):
            trend_score -= 0.1
        
        # Time influence (modified by drift)
        if 9 <= hour_of_day <= 11 or 19 <= hour_of_day <= 21:
            trend_score += self.current_weights['peak_hour_boost']
        elif 2 <= hour_of_day <= 6:
            trend_score -= abs(self.current_weights['peak_hour_boost'])
            
        # Weekend effect (can be reversed during drift)
        if day_of_week in [5, 6]:
            trend_score += self.current_weights['weekend_boost']
        
        # Time decay
        time_decay = np.exp(-time_since_base / self.current_weights['time_decay_factor'])
        trend_score += (time_decay - 0.5) * 0.3
        
        # Add noise (increased during drift)
        trend_score += np.random.normal(0, self.current_weights['noise_std'])
        
        # Normalize to [-1, 1] range
        return float(np.tanh(trend_score))
    
    def score_to_category(self, score: float) -> str:
        """Convert continuous score to categorical trend"""
        if score > 0.3:
            return "upward"
        elif score < -0.3:
            return "downward"
        else:
            return "neutral"

class DriftEvaluationTest:
    """Main class for conducting drift evaluation experiments"""
    
    def __init__(self, config: DriftTestConfig):
        self.config = config
        self.drift_simulator = ConceptDriftSimulator(config.embedding_dim)
        self.model_manager = None
        self.model = None
        self.records: List[SampleRecord] = []
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging for this test
        self.setup_logging()
        
    def setup_logging(self):
        """Setup test-specific logging"""
        log_file = self.output_dir / f"drift_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Starting drift evaluation test with config: {asdict(self.config)}")
    
    def initialize_system(self):
        """Initialize the ML system and create a trained model"""
        logger.info("Initializing TrendDetector system...")
        
        # Setup configuration
        app_config = AppConfig()
        app_config.model_settings.type = self.config.model_type
        app_config.model_settings.embedding_dim = self.config.embedding_dim
        
        # Initialize system
        gpu_manager = initialize_gpu(app_config.gpu_config.dict())
        self.model_manager = initialize_model_manager(app_config)
        
        # Create and train initial model
        self.model = self.model_manager.get_model(self.config.model_name, self.config.model_type)
        
        logger.info(f"Created {self.config.model_type} model: {self.config.model_name}")
    
    def train_initial_model(self, n_samples: int = 100):
        """Train the model with initial data"""
        logger.info(f"Training initial model with {n_samples} samples...")
        
        # Generate training data
        vectors = []
        targets = []
        timestamps = []
        
        base_time = datetime.now().timestamp()
        
        for i in range(n_samples):
            # Generate data with base parameters
            timestamp = base_time + i * 3600  # 1 hour intervals
            content_type = np.random.choice(["random", "viral", "trending"], p=[0.7, 0.15, 0.15])
            vector = self.drift_simulator.generate_embedding_vector(content_type)
            
            # Calculate true trend
            score = self.drift_simulator.calculate_true_trend_score(vector, timestamp, base_time)
            
            if self.config.model_type == "classification":
                target = self.drift_simulator.score_to_category(score)
            else:
                target = score
            
            vectors.append(vector)
            targets.append(target)
            timestamps.append(timestamp)
        
        # Train the model
        self.model.fit_initial(vectors, targets, timestamps)
        
        logger.info(f"Model trained successfully. Is fitted: {self.model.is_fitted}")
        
        # Log training distribution
        if self.config.model_type == "classification":
            unique, counts = np.unique(targets, return_counts=True)
            logger.info(f"Training distribution: {dict(zip(unique, counts))}")
        else:
            target_array = np.array(targets)
            logger.info(f"Training scores - Mean: {np.mean(target_array):.3f}, Std: {np.std(target_array):.3f}")
    
    def calculate_prediction_error(self, true_value: Any, predicted_value: Any) -> float:
        """Calculate prediction error based on model type"""
        if self.config.model_type == "classification":
            return 0.0 if true_value == predicted_value else 1.0
        else:  # regression
            return abs(float(true_value) - float(predicted_value))
    
    def run_inference_phase(self, n_samples: int, base_time: float, is_drift_period: bool = False):
        """Run inference phase with periodic feedback"""
        logger.info(f"Running inference phase: {n_samples} samples, drift_period={is_drift_period}")
        
        feedback_buffer = []
        
        for i in range(n_samples):
            # Generate sample
            timestamp = base_time + len(self.records) * 3600  # Progressive timestamps
            content_type = np.random.choice(["random", "viral", "trending"], p=[0.7, 0.15, 0.15])
            vector = self.drift_simulator.generate_embedding_vector(content_type)
            
            # Calculate true trend
            true_score = self.drift_simulator.calculate_true_trend_score(vector, timestamp, base_time)
            
            if self.config.model_type == "classification":
                true_trend = self.drift_simulator.score_to_category(true_score)
            else:
                true_trend = true_score
            
            # Make prediction
            result = self.model.predict(vector)
            
            if self.config.model_type == "classification":
                predicted_trend = result.predicted_trend
                confidence = result.confidence
            else:
                predicted_trend = result.predicted_score
                confidence = result.confidence
            
            # Calculate error
            error = self.calculate_prediction_error(true_trend, predicted_trend)
            
            # Create record
            record = SampleRecord(
                timestamp=timestamp,
                sample_index=len(self.records),
                embedding_vector=vector.tolist(),
                true_trend=true_trend,
                predicted_trend=predicted_trend,
                confidence=confidence,
                prediction_error=error,
                is_drift_period=is_drift_period,
                feedback_provided=False
            )
            
            self.records.append(record)
            feedback_buffer.append((vector, true_trend, predicted_trend, timestamp))
            
            # Provide feedback every N samples
            if len(feedback_buffer) >= self.config.feedback_frequency:
                logger.info(f"Providing feedback for {len(feedback_buffer)} samples at sample {record.sample_index}")
                
                drift_detected_any = False
                for fb_vector, fb_true, fb_pred, fb_timestamp in feedback_buffer:
                    drift_detected = self.model.learn(
                        features=fb_vector,
                        target=fb_true,
                        predicted_value=fb_pred,
                        timestamp=fb_timestamp
                    )
                    if drift_detected:
                        drift_detected_any = True
                
                # Mark records as having received feedback
                for j in range(len(feedback_buffer)):
                    self.records[-(j+1)].feedback_provided = True
                    self.records[-(j+1)].drift_detected = drift_detected_any
                
                if drift_detected_any:
                    logger.warning(f"Concept drift detected during feedback at sample {record.sample_index}")
                
                feedback_buffer.clear()
    
    def run_complete_experiment(self):
        """Run the complete drift evaluation experiment"""
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE DRIFT EVALUATION EXPERIMENT")
        logger.info("=" * 60)
        
        # Initialize system and train model
        self.initialize_system()
        self.train_initial_model(50)
        
        base_time = datetime.now().timestamp()
        
        # Phase 1: Initial run (no drift)
        logger.info("PHASE 1: Initial inference phase (no drift)")
        self.run_inference_phase(self.config.n_initial_samples, base_time, is_drift_period=False)
        
        # Phase 2: Introduce drift
        logger.info("PHASE 2: Introducing concept drift")
        self.drift_simulator.introduce_drift(self.config.drift_magnitude)
        
        # Phase 3: Drift period
        logger.info("PHASE 3: Inference during drift period")
        self.run_inference_phase(self.config.n_drift_samples, base_time, is_drift_period=True)
        
        logger.info("=" * 60)
        logger.info("DRIFT EVALUATION EXPERIMENT COMPLETED")
        logger.info(f"Total samples processed: {len(self.records)}")
        logger.info("=" * 60)
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experimental results"""
        logger.info("Analyzing experimental results...")
        
        df = pd.DataFrame([asdict(record) for record in self.records])
        
        # Calculate metrics
        pre_drift_records = df[~df['is_drift_period']]
        drift_records = df[df['is_drift_period']]
        
        analysis = {
            'total_samples': len(self.records),
            'pre_drift_samples': len(pre_drift_records),
            'drift_samples': len(drift_records),
            'pre_drift_error': {
                'mean': pre_drift_records['prediction_error'].mean(),
                'std': pre_drift_records['prediction_error'].std(),
                'max': pre_drift_records['prediction_error'].max()
            },
            'drift_error': {
                'mean': drift_records['prediction_error'].mean(),
                'std': drift_records['prediction_error'].std(),
                'max': drift_records['prediction_error'].max()
            },
            'drift_detected_count': sum(record.drift_detected for record in self.records),
            'feedback_provided_count': sum(record.feedback_provided for record in self.records)
        }
        
        # Calculate error change
        if len(pre_drift_records) > 0 and len(drift_records) > 0:
            error_increase = ((analysis['drift_error']['mean'] - analysis['pre_drift_error']['mean']) / 
                            analysis['pre_drift_error']['mean']) * 100
            analysis['error_increase_percent'] = error_increase
        
        # Log key findings
        logger.info("=" * 40)
        logger.info("ANALYSIS RESULTS:")
        logger.info(f"Pre-drift error: {analysis['pre_drift_error']['mean']:.4f} ± {analysis['pre_drift_error']['std']:.4f}")
        logger.info(f"Drift period error: {analysis['drift_error']['mean']:.4f} ± {analysis['drift_error']['std']:.4f}")
        if 'error_increase_percent' in analysis:
            logger.info(f"Error increase: {analysis['error_increase_percent']:.2f}%")
        logger.info(f"Drift detection events: {analysis['drift_detected_count']}")
        logger.info("=" * 40)
        
        return analysis
    
    def create_visualizations(self, analysis: Dict[str, Any]):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        # Prepare data
        df = pd.DataFrame([asdict(record) for record in self.records])
        df['time_hours'] = (df['timestamp'] - df['timestamp'].min()) / 3600
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Concept Drift Evaluation - {self.config.model_type.title()} Model', fontsize=16)
        
        # Plot 1: Prediction Error over Time
        ax1 = axes[0, 0]
        pre_drift = df[~df['is_drift_period']]
        drift_period = df[df['is_drift_period']]
        
        ax1.plot(pre_drift['time_hours'], pre_drift['prediction_error'], 'b-', alpha=0.7, label='Pre-drift')
        ax1.plot(drift_period['time_hours'], drift_period['prediction_error'], 'r-', alpha=0.7, label='Drift period')
        
        # Mark drift introduction point
        drift_start_time = df[df['is_drift_period']]['time_hours'].min() if len(drift_period) > 0 else None
        if drift_start_time:
            ax1.axvline(x=drift_start_time, color='orange', linestyle='--', linewidth=2, label='Drift introduced')
        
        # Mark feedback points
        feedback_points = df[df['feedback_provided']]
        if len(feedback_points) > 0:
            ax1.scatter(feedback_points['time_hours'], feedback_points['prediction_error'], 
                       c='green', marker='o', s=30, alpha=0.6, label='Feedback provided')
        
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Prediction Error')
        ax1.set_title('Prediction Error Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: True vs Predicted Trends
        ax2 = axes[0, 1]
        
        if self.config.model_type == "classification":
            # For classification, show trend categories over time
            trend_map = {'upward': 1, 'neutral': 0, 'downward': -1}
            df['true_numeric'] = df['true_trend'].map(trend_map)
            df['pred_numeric'] = df['predicted_trend'].map(trend_map)
            
            ax2.plot(df['time_hours'], df['true_numeric'], 'g-', linewidth=2, label='True trend', alpha=0.8)
            ax2.plot(df['time_hours'], df['pred_numeric'], 'b--', linewidth=1.5, label='Predicted trend', alpha=0.8)
            ax2.set_ylabel('Trend Category')
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(['Downward', 'Neutral', 'Upward'])
        else:
            # For regression, show continuous scores
            ax2.plot(df['time_hours'], df['true_trend'], 'g-', linewidth=2, label='True score', alpha=0.8)
            ax2.plot(df['time_hours'], df['predicted_trend'], 'b--', linewidth=1.5, label='Predicted score', alpha=0.8)
            ax2.set_ylabel('Trend Score')
        
        if drift_start_time:
            ax2.axvline(x=drift_start_time, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        
        ax2.set_xlabel('Time (hours)')
        ax2.set_title('True vs Predicted Trends')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rolling Error Statistics
        ax3 = axes[1, 0]
        window_size = 20
        df['rolling_error'] = df['prediction_error'].rolling(window=window_size, min_periods=1).mean()
        df['rolling_std'] = df['prediction_error'].rolling(window=window_size, min_periods=1).std()
        
        ax3.plot(df['time_hours'], df['rolling_error'], 'purple', linewidth=2, label=f'Rolling mean (window={window_size})')
        ax3.fill_between(df['time_hours'], 
                        df['rolling_error'] - df['rolling_std'],
                        df['rolling_error'] + df['rolling_std'],
                        alpha=0.2, color='purple', label='±1 std')
        
        if drift_start_time:
            ax3.axvline(x=drift_start_time, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Rolling Prediction Error')
        ax3.set_title('Rolling Error Statistics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Confidence vs Error
        ax4 = axes[1, 1]
        
        # Color points by drift period
        pre_drift_data = df[~df['is_drift_period']]
        drift_data = df[df['is_drift_period']]
        
        if len(pre_drift_data) > 0:
            ax4.scatter(pre_drift_data['confidence'], pre_drift_data['prediction_error'], 
                       c='blue', alpha=0.6, s=20, label='Pre-drift')
        
        if len(drift_data) > 0:
            ax4.scatter(drift_data['confidence'], drift_data['prediction_error'], 
                       c='red', alpha=0.6, s=20, label='Drift period')
        
        ax4.set_xlabel('Prediction Confidence')
        ax4.set_ylabel('Prediction Error')
        ax4.set_title('Confidence vs Error Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if self.config.save_plots:
            plot_file = self.output_dir / f"drift_analysis_{self.config.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {plot_file}")
        
        plt.show()
    
    def save_results(self, analysis: Dict[str, Any]):
        """Save experimental results to files"""
        if not self.config.save_data:
            return
            
        logger.info("Saving experimental results...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw data
        raw_data = {
            'config': asdict(self.config),
            'records': [asdict(record) for record in self.records],
            'analysis': analysis,
            'timestamp': timestamp
        }
        
        data_file = self.output_dir / f"drift_experiment_{self.config.model_type}_{timestamp}.json"
        with open(data_file, 'w') as f:
            json.dump(raw_data, f, indent=2, default=str)
        
        logger.info(f"Raw data saved to {data_file}")
        
        # Save analysis summary
        summary_file = self.output_dir / f"drift_summary_{self.config.model_type}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Analysis summary saved to {summary_file}")
        
        # Save CSV for further analysis
        df = pd.DataFrame([asdict(record) for record in self.records])
        csv_file = self.output_dir / f"drift_data_{self.config.model_type}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"CSV data saved to {csv_file}")

def run_drift_evaluation_test(model_type: str = "classification", 
                            n_initial: int = 200, 
                            n_drift: int = 300,
                            drift_magnitude: float = 0.5,
                            feedback_freq: int = 10):
    """Run a complete drift evaluation test"""
    
    config = DriftTestConfig(
        model_type=model_type,
        n_initial_samples=n_initial,
        n_drift_samples=n_drift,
        drift_magnitude=drift_magnitude,
        feedback_frequency=feedback_freq,
        output_dir=f"./drift_test_results_{model_type}"
    )
    
    test = DriftEvaluationTest(config)
    test.run_complete_experiment()
    analysis = test.analyze_results()
    test.create_visualizations(analysis)
    test.save_results(analysis)
    
    return test, analysis

if __name__ == "__main__":
    # Install matplotlib if not available
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Installing matplotlib for visualization...")
        os.system("pip install matplotlib pandas")
        import matplotlib.pyplot as plt
    
    # Run tests for both model types
    print("Running drift evaluation for classification model...")
    class_test, class_analysis = run_drift_evaluation_test("classification")
    
    print("\nRunning drift evaluation for regression model...")
    reg_test, reg_analysis = run_drift_evaluation_test("regression")
    
    print("\n" + "="*60)
    print("DRIFT EVALUATION TESTS COMPLETED")
    print("="*60)
    print(f"Classification error increase: {class_analysis.get('error_increase_percent', 'N/A'):.2f}%")
    print(f"Regression error increase: {reg_analysis.get('error_increase_percent', 'N/A'):.2f}%")