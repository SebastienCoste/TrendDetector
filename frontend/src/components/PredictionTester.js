import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Textarea } from "./ui/textarea";
import { Badge } from "./ui/badge";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Brain, Zap, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { apiService } from '../services/apiService';
import { useToast } from "../hooks/use-toast";

const PredictionTester = ({ models }) => {
  const [selectedModel, setSelectedModel] = useState('');
  const [inputVector, setInputVector] = useState('');
  const [sampleSize, setSampleSize] = useState(512);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  const generateRandomVector = () => {
    const vector = Array.from({ length: sampleSize }, () => (Math.random() - 0.5) * 2);
    setInputVector(JSON.stringify(vector, null, 2));
  };

  const generateTrendedVector = (trendType) => {
    const vector = Array.from({ length: sampleSize }, (_, i) => {
      let baseValue = Math.random() - 0.5;
      
      // Apply trend bias
      if (trendType === 'upward') {
        baseValue += Math.sin(i / sampleSize * Math.PI) * 0.5 + 0.2;
      } else if (trendType === 'downward') {
        baseValue -= Math.sin(i / sampleSize * Math.PI) * 0.5 + 0.2;
      }
      
      return baseValue;
    });
    
    setInputVector(JSON.stringify(vector, null, 2));
    toast({
      title: "Vector Generated",
      description: `Generated ${trendType} trending vector`,
    });
  };

  const makePrediction = async () => {
    if (!selectedModel || !inputVector.trim()) {
      toast({
        title: "Error",
        description: "Please select a model and provide input vector",
        variant: "destructive"
      });
      return;
    }

    setLoading(true);
    try {
      const vector = JSON.parse(inputVector);
      if (!Array.isArray(vector)) {
        throw new Error("Input must be an array of numbers");
      }

      const result = await apiService.makePrediction(vector, selectedModel);
      setPrediction(result);
      
      toast({
        title: "Prediction Complete",
        description: `Model: ${selectedModel}, Result: ${result.predicted_value}`,
      });
    } catch (error) {
      toast({
        title: "Prediction Failed",
        description: error.message || "Failed to make prediction",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const getTrendIcon = (trend) => {
    if (typeof trend === 'string') {
      switch (trend.toLowerCase()) {
        case 'upward': return <TrendingUp className="h-5 w-5 text-green-500" />;
        case 'downward': return <TrendingDown className="h-5 w-5 text-red-500" />;
        case 'neutral': return <Minus className="h-5 w-5 text-gray-500" />;
        default: return <Brain className="h-5 w-5" />;
      }
    } else {
      // Numeric trend score
      const score = parseFloat(trend);
      if (score > 0.3) return <TrendingUp className="h-5 w-5 text-green-500" />;
      if (score < -0.3) return <TrendingDown className="h-5 w-5 text-red-500" />;
      return <Minus className="h-5 w-5 text-gray-500" />;
    }
  };

  const availableModels = models.filter(m => m.is_loaded);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Single Prediction Test
          </CardTitle>
          <CardDescription>
            Test individual predictions with custom or generated vectors
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Model Selection */}
          <div className="space-y-2">
            <Label>Select Model</Label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger>
                <SelectValue placeholder="Choose a model..." />
              </SelectTrigger>
              <SelectContent>
                {availableModels.map(model => (
                  <SelectItem key={model.model_type} value={model.model_type}>
                    {model.model_name} ({model.model_type})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {availableModels.length === 0 && (
              <p className="text-xs text-red-500">No models are currently loaded</p>
            )}
          </div>

          {/* Vector Size */}
          <div className="space-y-2">
            <Label>Vector Dimension</Label>
            <Input
              type="number"
              value={sampleSize}
              onChange={(e) => setSampleSize(parseInt(e.target.value))}
              min={32}
              max={1024}
            />
          </div>

          {/* Quick Generate Buttons */}
          <div className="space-y-2">
            <Label>Quick Generate</Label>
            <div className="flex gap-2 flex-wrap">
              <Button variant="outline" size="sm" onClick={generateRandomVector}>
                Random Vector
              </Button>
              <Button variant="outline" size="sm" onClick={() => generateTrendedVector('upward')}>
                <TrendingUp className="h-4 w-4 mr-1" />
                Upward Trend
              </Button>
              <Button variant="outline" size="sm" onClick={() => generateTrendedVector('downward')}>
                <TrendingDown className="h-4 w-4 mr-1" />
                Downward Trend
              </Button>
              <Button variant="outline" size="sm" onClick={() => generateTrendedVector('neutral')}>
                <Minus className="h-4 w-4 mr-1" />
                Neutral
              </Button>
            </div>
          </div>

          {/* Vector Input */}
          <div className="space-y-2">
            <Label>Input Vector (JSON Array)</Label>
            <Textarea
              value={inputVector}
              onChange={(e) => setInputVector(e.target.value)}
              placeholder="[0.1, -0.2, 0.3, ...]"
              className="font-mono text-xs"
              rows={6}
            />
          </div>

          <Button 
            onClick={makePrediction} 
            disabled={loading || !selectedModel || availableModels.length === 0}
            className="w-full"
          >
            <Zap className="h-4 w-4 mr-2" />
            {loading ? "Predicting..." : "Make Prediction"}
          </Button>
        </CardContent>
      </Card>

      {/* Results Panel */}
      <Card>
        <CardHeader>
          <CardTitle>Prediction Result</CardTitle>
          <CardDescription>
            Model output and confidence metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          {prediction ? (
            <div className="space-y-4">
              {/* Main Result */}
              <div className="text-center">
                <div className="flex items-center justify-center mb-2">
                  {getTrendIcon(prediction.predicted_value)}
                </div>
                <h3 className="text-2xl font-bold">
                  {typeof prediction.predicted_value === 'string' 
                    ? prediction.predicted_value.toUpperCase()
                    : prediction.predicted_value.toFixed(4)}
                </h3>
                <p className="text-sm text-gray-600 capitalize">
                  {prediction.model_type} Prediction
                </p>
              </div>

              {/* Confidence */}
              <div className="text-center">
                <p className="text-sm font-medium text-gray-700 mb-1">Confidence</p>
                <Badge variant={prediction.confidence > 0.7 ? "default" : "secondary"}>
                  {(prediction.confidence * 100).toFixed(1)}%
                </Badge>
              </div>

              {/* Probabilities (Classification only) */}
              {prediction.probabilities && (
                <div className="space-y-2">
                  <p className="font-medium text-gray-700 text-sm">Class Probabilities</p>
                  <div className="space-y-1">
                    {Object.entries(prediction.probabilities).map(([trend, prob]) => (
                      <div key={trend} className="flex justify-between items-center text-sm">
                        <div className="flex items-center gap-2">
                          {getTrendIcon(trend)}
                          <span className="capitalize">{trend}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-20 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-500 h-2 rounded-full" 
                              style={{ width: `${prob * 100}%` }}
                            ></div>
                          </div>
                          <span className="w-12 text-right">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Metadata */}
              <div className="bg-gray-50 p-3 rounded text-xs space-y-1">
                <p><strong>Model Type:</strong> {prediction.model_type}</p>
                <p><strong>Timestamp:</strong> {new Date(prediction.timestamp).toLocaleString()}</p>
                {inputVector && (
                  <p><strong>Input Dimension:</strong> {JSON.parse(inputVector).length}</p>
                )}
              </div>

              {/* Interpretation */}
              <div className="bg-blue-50 p-3 rounded text-sm">
                <p className="font-medium text-blue-900 mb-1">Interpretation:</p>
                <p className="text-blue-800">
                  {prediction.model_type === 'classification' ? (
                    `The model predicts ${prediction.predicted_value} trend with ${(prediction.confidence * 100).toFixed(1)}% confidence.`
                  ) : (
                    `Trend score of ${prediction.predicted_value.toFixed(3)} indicates ${
                      prediction.predicted_value > 0.3 ? 'strong upward' :
                      prediction.predicted_value < -0.3 ? 'strong downward' : 'neutral'
                    } trending content.`
                  )}
                </p>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500 mb-2">No prediction results yet</p>
              <p className="text-xs text-gray-400">Select a model and input vector to get started</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default PredictionTester;