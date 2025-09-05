from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class BasePattern(str, Enum):
    LINEAR = "linear"
    SINUSOIDAL = "sinusoidal"
    EXPONENTIAL = "exponential"
    RANDOM_WALK = "random_walk"

class DriftTestConfig(BaseModel):
    num_requests: int = Field(default=100, ge=10, le=1000, description="Number of test requests")
    feedback_frequency: int = Field(default=10, ge=1, le=50, description="Feedback frequency")
    drift_point: int = Field(default=50, description="Point where drift is introduced")
    model_type: ModelType = Field(default=ModelType.REGRESSION, description="Model type to test")

class TemporalFactors(BaseModel):
    hourly: bool = Field(default=True, description="Include hourly patterns")
    daily: bool = Field(default=True, description="Include daily patterns") 
    weekly: bool = Field(default=False, description="Include weekly patterns")

class AlgorithmConfig(BaseModel):
    base_pattern: BasePattern = Field(default=BasePattern.SINUSOIDAL, description="Base pattern type")
    noise_level: float = Field(default=0.1, ge=0.0, le=1.0, description="Noise level")
    temporal_factors: TemporalFactors = Field(default_factory=TemporalFactors)
    velocity_influence: float = Field(default=0.3, ge=0.0, le=1.0, description="Velocity influence")
    embedding_correlation: float = Field(default=0.7, ge=0.0, le=1.0, description="Embedding correlation")

class VectorGenerationConfig(BaseModel):
    trend_score: float = Field(ge=-1.0, le=1.0, description="Target trend score")
    algorithm_params: AlgorithmConfig = Field(description="Algorithm parameters")
    timestamp: Optional[datetime] = Field(default=None, description="Generation timestamp")
    embedding_dim: int = Field(default=512, description="Embedding dimension")

class DriftTestResult(BaseModel):
    request_id: int
    timestamp: datetime
    embedding_vector: List[float]
    expected_trend: Any  # float for regression, str for classification
    predicted_trend: Any
    confidence: float
    absolute_error: float
    is_drift_period: bool
    feedback_provided: bool
    drift_detected: bool = False

class VectorGenerationResult(BaseModel):
    vector: List[float]
    expected_trend: Any
    timestamp: datetime
    algorithm_used: str

class PredictionResult(BaseModel):
    predicted_value: Any
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    model_type: str
    timestamp: datetime

class TestStatus(BaseModel):
    test_id: str
    status: str  # "running", "completed", "error"
    progress: float  # 0.0 to 1.0
    current_request: int
    total_requests: int
    metrics: Dict[str, Any]

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    is_loaded: bool
    version: str
    last_updated: Optional[datetime] = None