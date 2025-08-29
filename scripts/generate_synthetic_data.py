#!/usr/bin/env python3
"""
Synthetic Data Generator for Trending Content Detection System

Generates realistic test data with temporal patterns for model training and evaluation.
"""

import numpy as np
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
from pathlib import Path

class SyntheticDataGenerator:
    """Generate synthetic data with temporal trend patterns"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed

    def generate_embedding_vector(self, content_type: str = "random", dim: int = 512) -> np.ndarray:
        """Generate synthetic embedding vector"""
        if content_type == "random":
            return np.random.randn(dim)
        elif content_type == "viral":
            # Viral content tends to have higher magnitude in certain dimensions
            vector = np.random.randn(dim)
            # Amplify certain dimensions for viral content
            viral_dims = np.random.choice(dim, size=dim//10, replace=False)
            vector[viral_dims] *= 2.0
            return vector
        elif content_type == "boring":
            # Trending content has moderate amplification
            vector = np.random.randn(dim)
            trending_dims = np.random.choice(dim, size=dim//20, replace=False)
            vector[trending_dims] *= 0.1
            return vector
        else:
            return np.random.randn(dim)


    def calculate_trend_from_vector_and_time(self,
                                             vector: np.ndarray,
                                             timestamp: float,
                                             base_time: float,
                                             threshold: float) -> str:
        """Alternative trend calculation based on 10 hour/day-dependent vector values"""

        # Time-based features
        hour_of_day = int((timestamp % (24 * 3600)) / 3600)  # 0-23
        day_of_week = int(timestamp // (24 * 3600)) % 7      # 0-6

        # Deterministically select 10 indices based on hour and day
        dim = vector.shape[0]
        indices = [((hour_of_day * 31 + day_of_week * 17 + i * 13) % dim) for i in range(10)]

        selected_values = vector[indices]
        mean_value = np.mean(selected_values)

        if mean_value > threshold:
            return "upward"
        elif mean_value < -1 * threshold:
            return "downward"
        else:
            return "neutral"


    def calculate_trend_from_vector_and_time_complicated(self,
                                           vector: np.ndarray,
                                           timestamp: float,
                                           base_time: float) -> str:
        """Calculate trend based on vector characteristics and time"""

        # Vector-based features
        vector_sum = np.sum(vector)
        vector_magnitude = np.linalg.norm(vector)
        vector_variance = np.var(vector)

        # Time-based features
        time_since_base = timestamp - base_time
        hour_of_day = (timestamp % (24 * 3600)) / 3600  # Hour in 0-24
        day_of_week = int(timestamp // (24 * 3600)) % 7  # Day 0-6

        # Calculate trend score based on multiple factors
        trend_score = 0.0

        # Vector influence (40% of decision)
        if vector_sum > 5.0:
            trend_score += 0.2
        elif vector_sum < -5.0:
            trend_score -= 0.2

        if vector_magnitude > 25.0:
            trend_score += 0.1
        elif vector_magnitude < 15.0:
            trend_score -= 0.1

        if vector_variance > 1.5:
            trend_score += 0.1
        elif vector_variance < 0.5:
            trend_score -= 0.1

        # Time influence (30% of decision)
        # Peak hours (9-11, 19-21) favor upward trends
        if 9 <= hour_of_day <= 11 or 19 <= hour_of_day <= 21:
            trend_score += 0.15
        # Off-peak hours favor downward trends
        elif 2 <= hour_of_day <= 6:
            trend_score -= 0.15

        # Weekends slightly favor upward trends
        if day_of_week in [5, 6]:  # Saturday, Sunday
            trend_score += 0.1

        # Long-term time trend (30% of decision)
        # Content gets less trending over time (decay effect)
        time_decay = np.exp(-time_since_base / (7 * 24 * 3600))  # 7-day decay
        trend_score += (time_decay - 0.5) * 0.3

        # Add some randomness for realism
        trend_score += np.random.normal(0, 0.1)

        # Convert to categorical trend
        if trend_score > 0.15:
            return "upward"
        elif trend_score < -0.15:
            return "downward"
        else:
            return "neutral"

    def generate_velocity_features(self,
                                 trend: str,
                                 timestamp: float,
                                 noise_level: float = 0.1) -> Dict[str, float]:
        """Generate velocity features consistent with trend"""

        base_velocities = {
            "upward": {
                'download_velocity_1h': np.random.exponential(20),
                'download_velocity_24h': np.random.exponential(15),
                'like_velocity_1h': np.random.exponential(8),
                'like_velocity_24h': np.random.exponential(6),
                'dislike_velocity_1h': np.random.exponential(1),
                'dislike_velocity_24h': np.random.exponential(1),
                'rating_velocity_1h': np.random.normal(0.1, 0.05),
                'rating_velocity_24h': np.random.normal(0.08, 0.03)
            },
            "downward": {
                'download_velocity_1h': np.random.exponential(2),
                'download_velocity_24h': np.random.exponential(3),
                'like_velocity_1h': np.random.exponential(1),
                'like_velocity_24h': np.random.exponential(2),
                'dislike_velocity_1h': np.random.exponential(5),
                'dislike_velocity_24h': np.random.exponential(4),
                'rating_velocity_1h': np.random.normal(-0.1, 0.05),
                'rating_velocity_24h': np.random.normal(-0.08, 0.03)
            },
            "neutral": {
                'download_velocity_1h': np.random.exponential(5),
                'download_velocity_24h': np.random.exponential(5),
                'like_velocity_1h': np.random.exponential(3),
                'like_velocity_24h': np.random.exponential(3),
                'dislike_velocity_1h': np.random.exponential(2),
                'dislike_velocity_24h': np.random.exponential(2),
                'rating_velocity_1h': np.random.normal(0, 0.02),
                'rating_velocity_24h': np.random.normal(0, 0.02)
            }
        }

        # Get base velocities for trend
        velocities = base_velocities[trend].copy()

        # Add time-based variations
        hour_of_day = (timestamp % (24 * 3600)) / 3600

        # Peak hours increase velocities
        if 9 <= hour_of_day <= 11 or 19 <= hour_of_day <= 21:
            for key in velocities:
                if 'velocity' in key:
                    velocities[key] *= 1.3

        # Add noise
        for key in velocities:
            noise = np.random.normal(1, noise_level)
            velocities[key] *= max(0.1, noise)  # Prevent negative velocities

        return velocities

    def generate_dataset(self,
                        n_samples: int = 1000,
                        time_span_days: int = 30,
                        embedding_dim: int = 512,
                        threshold: float = 0.2,
                        start_date: datetime = None) -> Tuple[List[np.ndarray], List[str], List[float], List[Dict[str, float]]]:
        """Generate complete synthetic dataset"""

        if start_date is None:
            start_date = datetime.now() - timedelta(days=time_span_days)

        base_timestamp = start_date.timestamp()
        end_timestamp = (start_date + timedelta(days=time_span_days)).timestamp()

        vectors = []
        trends = []
        timestamps = []
        velocity_features_list = []

        print(f"Generating {n_samples} samples over {time_span_days} days starting {start_date}...")

        for i in range(n_samples):
            # Generate timestamp (uniformly distributed over time span)
            timestamp = base_timestamp + (end_timestamp - base_timestamp) * (i / n_samples)

            # Add some randomness to timestamps
            timestamp += np.random.normal(0, 7200) - 3600  # ±1 hour noise

            # Generate embedding vector
            content_type = np.random.choice(["random", "viral", "boring"], p=[0.7, 0.15, 0.15])
            vector = self.generate_embedding_vector(content_type, embedding_dim)

            # Calculate trend based on vector and time
            trend = self.calculate_trend_from_vector_and_time(vector, timestamp, base_timestamp, threshold)

            # Generate consistent velocity features
            velocity_features = self.generate_velocity_features(trend, timestamp)

            vectors.append(vector)
            trends.append(trend)
            timestamps.append(timestamp)
            velocity_features_list.append(velocity_features)

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} samples")

        # Print distribution
        trend_counts = {trend: trends.count(trend) for trend in ["upward", "downward", "neutral"]}
        print(f"Trend distribution: {trend_counts}")

        return vectors, trends, timestamps, velocity_features_list

    def save_dataset(self,
                    vectors: List[np.ndarray],
                    trends: List[str],
                    timestamps: List[float],
                    velocity_features_list: List[Dict[str, float]],
                    output_path: str) -> None:
        """Save dataset to files"""

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save vectors
        np.save(output_dir / "vectors.npy", np.array(vectors))

        # Save other data as JSON
        data = {
            "trends": trends,
            "timestamps": timestamps,
            "velocity_features": velocity_features_list,
            "metadata": {
                "n_samples": len(vectors),
                "embedding_dim": vectors[0].shape[0],
                "generation_time": datetime.now().isoformat(),
                "seed": self.seed
            }
        }

        with open(output_dir / "data.json", "w") as f:
            json.dump(data, f, indent=2)

        print(f"Dataset saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for trend detection")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--days", type=int, default=30, help="Time span in days")
    parser.add_argument("--output", type=str, default="./test_data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threshold", type=float, default=0.3, help="absolute  threshold of normal distribution defining upward/downward")

    args = parser.parse_args()

    generator = SyntheticDataGenerator(seed=args.seed)
    vectors, trends, timestamps, velocity_features_list = generator.generate_dataset(
        n_samples=args.samples,
        time_span_days=args.days,
        threshold=args.threshold,
    )

    generator.save_dataset(vectors, trends, timestamps, velocity_features_list, args.output)

if __name__ == "__main__":
    main()