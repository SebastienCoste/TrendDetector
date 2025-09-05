# TrendDetector Drift Test Workflow Demo

## 🎯 Complete Drift Testing Demonstration

This guide provides a comprehensive workflow for demonstrating the TrendDetector's concept drift evaluation capabilities through both the UI and programmatic interfaces.

## 🚀 Quick Start Demo (UI)

### Step 1: Access the TrendDetector UI
```
1. Navigate to: http://localhost:3000
2. Wait for both models to load (should show "Models Loaded: 2/2")
3. Click on the "Drift Testing" tab
```

### Step 2: Configure Drift Test
```
Recommended Settings for Demo:
- Model Type: "regression" (for continuous visualization)
- Total Requests: 100
- Drift Point: 50 (introduces drift halfway through)
- Feedback Frequency: 10 (provides feedback every 10 samples)
```

### Step 3: Start and Monitor Test
```
1. Click "Start Drift Test" 
2. Observe real-time progress bar
3. Watch the test ID appear in "Active Tests"
4. Monitor test status updates every 2 seconds
```

### Step 4: View Visualization
```
Once test completes, explore the visualization tabs:
- "Prediction Error": Shows error spike at drift point
- "True vs Predicted": Displays model adaptation
- "Confidence": Shows confidence changes during drift
```

## 📊 Expected Results

### Pre-Drift Period (Requests 1-50)
- **Low Error**: Typically 0.1-0.3 range
- **Stable Predictions**: Model performs well on original data distribution
- **High Confidence**: Model is confident in predictions

### Drift Introduction (Request 50)
- **Error Spike**: Sudden increase in prediction error
- **Divergence**: True vs predicted values start to diverge
- **Confidence Drop**: Model becomes less confident

### Adaptation Period (Requests 51-100)
- **Error Recovery**: Error gradually decreases as model adapts
- **Convergence**: Predictions align better with new distribution
- **Confidence Recovery**: Model regains confidence

## 🔧 Programmatic Demo Script

Create and run this script for automated demonstration:

```python
#!/usr/bin/env python3
"""
TrendDetector Drift Test Demo Script
Demonstrates complete drift testing workflow programmatically
"""

import asyncio
import time
from src.services.apiService import apiService

async def run_drift_demo():
    print("🎯 TrendDetector Drift Test Demo Starting...")
    
    # Step 1: Check system health
    print("\n1. Checking System Health...")
    models = await apiService.getModels()
    print(f"   ✅ Found {len(models)} models")
    for model in models:
        status = "✅ Loaded" if model.is_loaded else "❌ Not Loaded"
        print(f"   {status}: {model.model_name} ({model.model_type})")
    
    # Step 2: Start drift test
    print("\n2. Starting Drift Test...")
    config = {
        "num_requests": 100,
        "feedback_frequency": 10,
        "drift_point": 50,
        "model_type": "regression"
    }
    
    result = await apiService.startDriftTest(config)
    test_id = result["test_id"]
    print(f"   ✅ Test started: {test_id[:8]}...")
    
    # Step 3: Monitor progress
    print("\n3. Monitoring Test Progress...")
    while True:
        status = await apiService.getDriftTestStatus(test_id)
        progress = status["progress"] * 100
        current = status["current_request"]
        total = status["total_requests"]
        
        print(f"   Progress: {progress:.1f}% ({current}/{total}) - Status: {status['status']}")
        
        if status["status"] in ["completed", "error"]:
            break
            
        await asyncio.sleep(2)
    
    # Step 4: Analyze results
    print("\n4. Analyzing Results...")
    results = await apiService.getDriftTestResults(test_id)
    
    # Calculate metrics
    pre_drift = [r for r in results if not r["is_drift_period"]]
    drift_period = [r for r in results if r["is_drift_period"]]
    
    pre_error = sum(r["absolute_error"] for r in pre_drift) / len(pre_drift)
    drift_error = sum(r["absolute_error"] for r in drift_period) / len(drift_period)
    error_increase = ((drift_error - pre_error) / pre_error) * 100
    
    print(f"   📊 Pre-drift Error: {pre_error:.4f}")
    print(f"   📊 Drift Period Error: {drift_error:.4f}")
    print(f"   📊 Error Increase: {error_increase:.1f}%")
    print(f"   📊 Total Samples: {len(results)}")
    print(f"   📊 Feedback Events: {sum(1 for r in results if r['feedback_provided'])}")
    
    # Step 5: Cleanup
    print(f"\n5. Cleaning up test {test_id[:8]}...")
    await apiService.cleanupDriftTest(test_id)
    print("   ✅ Test cleaned up successfully")
    
    print("\n🎉 Drift Test Demo Complete!")
    print("   Check the UI at http://localhost:3000 to see visualization")

if __name__ == "__main__":
    asyncio.run(run_drift_demo())
```

## 🎨 UI Demonstration Steps

### Visual Demo Walkthrough

1. **System Overview**
   ```
   - Navigate to Models tab
   - Show system health (Service: Healthy, GPU: CPU Only, Models: 2/2 loaded)
   - Display both models with 100% readiness
   ```

2. **Vector Generation Demo**
   ```
   - Go to Vector Generator tab
   - Adjust trend score slider to 0.7 (upward trend)
   - Select "sinusoidal" pattern
   - Enable hourly and daily temporal factors
   - Click Generate Vector
   - Show vector statistics and preview
   ```

3. **Single Prediction Demo**
   ```
   - Go to Predictions tab
   - Select "regression" model
   - Click "Upward Trend" quick generate
   - Click "Make Prediction"
   - Show result: positive trend score with confidence
   ```

4. **Complete Drift Test Demo**
   ```
   - Go to Drift Testing tab
   - Configure: 100 requests, drift at 50, feedback every 10
   - Start test and monitor real-time progress
   - Watch error visualization during drift
   - Show adaptation in True vs Predicted chart
   - Review final metrics and summary
   ```

## 📈 Understanding the Visualizations

### Prediction Error Chart
- **X-axis**: Request number (1-100)
- **Y-axis**: Absolute prediction error
- **Red dashed line**: Drift introduction point
- **Blue line**: Error over time
- **Expected pattern**: Low → Spike → Gradual decrease

### True vs Predicted Chart
- **Green line**: Expected/true values
- **Blue line**: Model predictions
- **Divergence**: Shows when model struggles with new data
- **Convergence**: Shows model adaptation

### Confidence Chart
- **Purple line**: Model confidence (0-1 scale)
- **Expected pattern**: High → Drop → Recovery
- **Indicates**: Model's certainty in its predictions

## 🔍 Key Metrics to Highlight

1. **Error Change Percentage**: How much error increased during drift
2. **Feedback Events**: Number of learning updates provided
3. **Drift Detections**: Automatic drift detection triggers
4. **Adaptation Speed**: How quickly error returns to baseline

## 🎯 Demo Scenarios

### Scenario 1: Light Drift (Quick Demo)
```
- Requests: 50
- Drift Point: 25
- Expected: Small error increase, quick recovery
- Duration: ~2 minutes
```

### Scenario 2: Heavy Drift (Detailed Demo)
```
- Requests: 200
- Drift Point: 100
- Feedback Frequency: 15
- Expected: Significant error increase, slower recovery
- Duration: ~5 minutes
```

### Scenario 3: Classification Demo
```
- Model Type: Classification
- Requests: 80
- Drift Point: 40
- Shows: Category prediction changes and adaptation
```

## 🚨 Troubleshooting

### If Test Doesn't Start
1. Check both models are loaded (Models tab)
2. Verify service health (green checkmark)
3. Ensure valid configuration values

### If Visualization Doesn't Show
1. Wait for test completion
2. Refresh the page
3. Check active tests list

### If Error Rates Seem Wrong
1. This is normal variation in synthetic data
2. Focus on the trend pattern rather than absolute values
3. Multiple runs will show consistent patterns

## 🎪 Presentation Tips

1. **Start with System Health**: Show robust, production-ready system
2. **Build Up Complexity**: Vector → Prediction → Drift Test
3. **Emphasize Real-time**: Show live progress updates
4. **Explain Visualizations**: Walk through what each chart means
5. **Highlight Adaptation**: Show how the model learns and recovers

## 📝 Expected Demo Outcomes

✅ **System demonstrates**: Concept drift detection and adaptation  
✅ **UI shows**: Professional, real-time monitoring interface  
✅ **Models prove**: Dual-model architecture works for both classification and regression  
✅ **Visualizations reveal**: Clear patterns of drift introduction and recovery  
✅ **Metrics confirm**: Quantitative evidence of model adaptation  

---

**🎉 This demo showcases a production-ready ML system with concept drift handling capabilities!**