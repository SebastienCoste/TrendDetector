# Trending Content Detection System

A real-time machine learning service that predicts content trends using adaptive algorithms with concept drift detection. Now supports both classification and regression models through a unified interface.

## Features

- **Dual-Model Support** - Both classification (upward/downward/neutral) and regression (continuous score -1 to 1)
- **Real-time trend prediction** with configurable model types
- **Continuous learning** with adaptive model updates
- **Concept drift detection** using adaptive algorithms
- **KServe V2 API compliance** for production compatibility
- **GPU acceleration** with CPU fallback
- **Model versioning** and checkpoint management
- **Embedding-only input** (velocity features used only for target generation)
- **Comprehensive evaluation metrics** (MAE, RMSE, R², accuracy, precision, recall, F1)

## System Architecture

The system implements a dual-model architecture with the following components:

- **FastAPI Application**: KServe V2 compliant REST API with dynamic model routing
- **Model Interface Layer**: Unified interface for both classification and regression
- **Adaptive Random Forest Models**: Separate classifiers and regressors using River
- **Dynamic Trend Memory**: GPU-accelerated similarity matching
- **Concept Drift Detection**: Automated model adaptation
- **Model Manager**: Version control and persistence for both model types
- **Evaluation System**: Configurable metrics for both model types

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Test Data

```bash
python scripts/generate_synthetic_data.py --samples 1000 --output ./test_data
```

### 3. Start the Service

```bash
python -m src.main
```

### 4. Test the API

```bash
# Check health
curl http://localhost:8080/health

# Get classification model metadata
curl "http://localhost:8080/v2/models/trend_classifier?model_type=classification"

# Get regression model metadata
curl "http://localhost:8080/v2/models/trend_regressor?model_type=regression"

# Make a classification prediction
curl -X POST "http://localhost:8080/v2/models/trend_classifier/infer?model_type=classification" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "embedding_vector",
        "shape": [512],
        "datatype": "FP32",
        "data": [/* 512 float values */]
      }
    ],
    "parameters": {"model_type": "classification"}
  }'

# Make a regression prediction
curl -X POST "http://localhost:8080/v2/models/trend_regressor/infer?model_type=regression" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "embedding_vector",
        "shape": [512],
        "datatype": "FP32",
        "data": [/* 512 float values */]
      }
    ],
    "parameters": {"model_type": "regression"}
  }'
```

## API Endpoints

### KServe V2 Inference API

- `GET /v2/models/{model_name}` - Get model metadata
- `POST /v2/models/{model_name}/infer` - Make predictions
- `POST /v2/models/{model_name}/update` - Update with feedback
- `GET /v2/models/{model_name}/stats` - Get model statistics

### Health and Status

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /docs` - API documentation

## Configuration

The system uses YAML configuration files located in `config/config.yaml`:

```yaml
server_config:
  host: "0.0.0.0"
  port: 8080

model_settings:
  type: "classification"  # or "regression"
  n_trees: 10
  drift_threshold: 0.01
  embedding_dim: 512
  output_range: [-1, 1]  # for regression models

evaluation_config:
  regression_metrics: ["mae", "rmse", "r2"]
  primary_regression_metric: "mae"
  classification_metrics: ["accuracy", "precision", "recall", "f1"]
  primary_classification_metric: "accuracy"

gpu_config:
  enabled: true
  fallback_cpu: true
```

## Data Format

### Input Format (Inference)

```json
{
  "inputs": [
    {
      "name": "embedding_vector",
      "shape": [512],
      "datatype": "FP32", 
      "data": [0.1, 0.2, ...]
    },
    {
      "name": "velocity_features",
      "shape": [8],
      "datatype": "FP32",
      "data": [25.0, 120.5, 5.8, 22.3, -2.1, -8.7, 0.02, 0.15]
    }
  ]
}
```

### Output Format

```json
{
  "model_name": "trend_classifier",
  "model_version": "v1",
  "outputs": [
    {
      "name": "predicted_trend",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["upward"]
    },
    {
      "name": "confidence", 
      "shape": [1],
      "datatype": "FP32",
      "data": [0.85]
    }
  ]
}
```

## Project Structure

```
trending-content-detection/
├── config/
│   └── config.yaml
├── src/
│   ├── main.py
│   ├── api/
│   │   └── v2/
│   │       ├── models.py
│   │       ├── inference.py
│   │       ├── update.py
│   │       └── stats.py
│   ├── ml/
│   │   ├── adaptive_classifier.py
│   │   └── trend_memory.py
│   ├── core/
│   │   ├── config.py
│   │   ├── gpu_utils.py
│   │   └── model_manager.py
│   └── utils/
│       └── validators.py
├── scripts/
│   └── generate_synthetic_data.py
├── models/
├── logs/
└── test_data/
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
flake8 src/
```

### Performance Testing

```bash
python scripts/generate_synthetic_data.py --samples 10000
python -m pytest tests/test_performance.py
```

## Monitoring

The system provides comprehensive monitoring through:

- **Structured logging** with configurable levels
- **Performance metrics** via `/v2/models/{model_name}/stats`
- **Health checks** via `/health` endpoint
- **GPU memory monitoring** when enabled

## Deployment

For production deployment:

1. Configure GPU settings in `config.yaml`
2. Set up proper logging directories
3. Configure CORS origins if needed
4. Use a production ASGI server like uvicorn or gunicorn

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8080 --workers 4
```

## License

This project implements the technical specifications from the Trending Content Detection System documentation.