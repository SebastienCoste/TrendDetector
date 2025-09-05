from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum

class DataType(str, Enum):
    """KServe V2 supported data types"""
    BOOL = "BOOL"
    UINT8 = "UINT8"
    UINT16 = "UINT16"
    UINT32 = "UINT32"
    UINT64 = "UINT64"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FP16 = "FP16"
    FP32 = "FP32"
    FP64 = "FP64"
    BYTES = "BYTES"

class InferTensor(BaseModel):
    """KServe V2 inference tensor"""
    name: str = Field(..., description="Name of the tensor")
    shape: List[int] = Field(..., description="Shape of the tensor")
    datatype: DataType = Field(..., description="Data type of the tensor")
    data: List[Union[str, int, float]] = Field(..., description="Tensor data")

    @validator('shape')
    def validate_shape(cls, v):
        if not v or any(dim <= 0 for dim in v):
            raise ValueError("Shape dimensions must be positive")
        return v

class InferenceRequest(BaseModel):
    """KServe V2 inference request"""
    inputs: List[InferTensor] = Field(..., description="Input tensors")
    outputs: Optional[List[Dict[str, Any]]] = Field(None, description="Output specifications")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Request parameters")
    
    def get_model_type(self) -> Optional[str]:
        """Extract model type from parameters if provided"""
        if self.parameters and 'model_type' in self.parameters:
            return self.parameters['model_type']
        return None

    @validator('inputs')
    def validate_inputs(cls, v):
        if not v:
            raise ValueError("At least one input tensor is required")
        return v

class InferenceResponse(BaseModel):
    """KServe V2 inference response"""
    model_name: str = Field(..., description="Name of the model")
    model_version: Optional[str] = Field(None, description="Version of the model")
    outputs: List[InferTensor] = Field(..., description="Output tensors")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Response parameters")

class ModelMetadata(BaseModel):
    """KServe V2 model metadata"""
    name: str = Field(..., description="Model name")
    versions: Optional[List[str]] = Field(None, description="Available versions")
    platform: str = Field(..., description="Model platform")
    inputs: List[Dict[str, Any]] = Field(..., description="Input specifications")
    outputs: List[Dict[str, Any]] = Field(..., description="Output specifications")

class UpdateRequest(BaseModel):
    """Custom update request for trend feedback"""
    updates: List[Dict[str, Any]] = Field(..., description="List of trend updates")
    model_type: Optional[str] = Field(None, description="Model type override")

    @validator('updates')
    def validate_updates(cls, v):
        # Basic validation - specific fields will be validated per model type
        required_fields = ['embedding_vector']
        for update in v:
            for field in required_fields:
                if field not in update:
                    raise ValueError(f"Missing required field: {field}")
            # Check for either trend or score
            if 'actual_trend' not in update and 'actual_score' not in update:
                raise ValueError("Missing target field: either 'actual_trend' or 'actual_score' required")
        return v

class UpdateResponse(BaseModel):
    """Custom update response"""
    processed_updates: int = Field(..., description="Number of processed updates")
    drift_detected: bool = Field(..., description="Whether concept drift was detected")
    model_version: str = Field(..., description="Current model version")
    message: str = Field(..., description="Response message")

class StatsResponse(BaseModel):
    """Model statistics response"""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    statistics: Dict[str, Any] = Field(..., description="Model statistics")

class ErrorResponse(BaseModel):
    """Error response format"""
    error: Dict[str, Any] = Field(..., description="Error details")