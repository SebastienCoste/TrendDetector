import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Slider } from "./ui/slider";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";
import { Textarea } from "./ui/textarea";
import { Badge } from "./ui/badge";
import { Wand2, Copy, Download } from "lucide-react";
import { apiService } from '../services/apiService';
import { useToast } from "../hooks/use-toast";

const VectorGenerator = () => {
  const [algorithms, setAlgorithms] = useState(null);
  const [config, setConfig] = useState({
    trend_score: 0.5,
    algorithm_params: {
      base_pattern: 'sinusoidal',
      noise_level: 0.1,
      temporal_factors: {
        hourly: true,
        daily: true,
        weekly: false
      },
      velocity_influence: 0.3,
      embedding_correlation: 0.7
    },
    embedding_dim: 512
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    fetchAlgorithms();
  }, []);

  const fetchAlgorithms = async () => {
    try {
      const algorithmsData = await apiService.getAlgorithms();
      setAlgorithms(algorithmsData);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch algorithm information",
        variant: "destructive"
      });
    }
  };

  const generateVector = async () => {
    setLoading(true);
    try {
      const result = await apiService.generateVector(config);
      setResult(result);
      toast({
        title: "Success",
        description: `Generated ${result.vector.length}-dimensional vector`,
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to generate vector",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const copyVector = () => {
    if (result) {
      navigator.clipboard.writeText(JSON.stringify(result.vector));
      toast({
        title: "Copied",
        description: "Vector copied to clipboard",
      });
    }
  };

  const downloadVector = () => {
    if (result) {
      const dataStr = JSON.stringify(result, null, 2);
      const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
      const exportFileDefaultName = `vector_${Date.now()}.json`;
      
      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', exportFileDefaultName);
      linkElement.click();
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Configuration Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wand2 className="h-5 w-5" />
            Vector Generation
          </CardTitle>
          <CardDescription>
            Configure parameters to generate embedding vectors with specific characteristics
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Target Trend Score */}
          <div className="space-y-2">
            <Label>Target Trend Score: {config.trend_score.toFixed(2)}</Label>
            <Slider
              value={[config.trend_score]}
              onValueChange={([value]) => setConfig(prev => ({ ...prev, trend_score: value }))}
              min={-1}
              max={1}
              step={0.1}
              className="w-full"
            />
            <p className="text-xs text-gray-500">-1 (strong downward) to 1 (strong upward)</p>
          </div>

          {/* Base Pattern */}
          <div className="space-y-2">
            <Label>Base Pattern</Label>
            <Select
              value={config.algorithm_params.base_pattern}
              onValueChange={(value) => setConfig(prev => ({
                ...prev,
                algorithm_params: { ...prev.algorithm_params, base_pattern: value }
              }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {algorithms?.base_patterns?.map(pattern => (
                  <SelectItem key={pattern} value={pattern}>
                    {pattern.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Noise Level */}
          <div className="space-y-2">
            <Label>Noise Level: {config.algorithm_params.noise_level.toFixed(2)}</Label>
            <Slider
              value={[config.algorithm_params.noise_level]}
              onValueChange={([value]) => setConfig(prev => ({
                ...prev,
                algorithm_params: { ...prev.algorithm_params, noise_level: value }
              }))}
              min={0}
              max={1}
              step={0.05}
            />
          </div>

          {/* Embedding Correlation */}
          <div className="space-y-2">
            <Label>Embedding Correlation: {config.algorithm_params.embedding_correlation.toFixed(2)}</Label>
            <Slider
              value={[config.algorithm_params.embedding_correlation]}
              onValueChange={([value]) => setConfig(prev => ({
                ...prev,
                algorithm_params: { ...prev.algorithm_params, embedding_correlation: value }
              }))}
              min={0}
              max={1}
              step={0.05}
            />
            <p className="text-xs text-gray-500">How strongly the vector correlates with the target trend</p>
          </div>

          {/* Temporal Factors */}
          <div className="space-y-3">
            <Label>Temporal Factors</Label>
            <div className="flex items-center space-x-2">
              <Switch
                id="hourly"
                checked={config.algorithm_params.temporal_factors.hourly}
                onCheckedChange={(checked) => setConfig(prev => ({
                  ...prev,
                  algorithm_params: {
                    ...prev.algorithm_params,
                    temporal_factors: { ...prev.algorithm_params.temporal_factors, hourly: checked }
                  }
                }))}
              />
              <Label htmlFor="hourly">Hourly patterns</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="daily"
                checked={config.algorithm_params.temporal_factors.daily}
                onCheckedChange={(checked) => setConfig(prev => ({
                  ...prev,
                  algorithm_params: {
                    ...prev.algorithm_params,
                    temporal_factors: { ...prev.algorithm_params.temporal_factors, daily: checked }
                  }
                }))}
              />
              <Label htmlFor="daily">Daily patterns</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="weekly"
                checked={config.algorithm_params.temporal_factors.weekly}
                onCheckedChange={(checked) => setConfig(prev => ({
                  ...prev,
                  algorithm_params: {
                    ...prev.algorithm_params,
                    temporal_factors: { ...prev.algorithm_params.temporal_factors, weekly: checked }
                  }
                }))}
              />
              <Label htmlFor="weekly">Weekly patterns</Label>
            </div>
          </div>

          {/* Embedding Dimension */}
          <div className="space-y-2">
            <Label>Embedding Dimension</Label>
            <Input
              type="number"
              value={config.embedding_dim}
              onChange={(e) => setConfig(prev => ({ ...prev, embedding_dim: parseInt(e.target.value) }))}
              min={32}
              max={1024}
            />
          </div>

          <Button onClick={generateVector} disabled={loading} className="w-full">
            {loading ? "Generating..." : "Generate Vector"}
          </Button>
        </CardContent>
      </Card>

      {/* Results Panel */}
      <Card>
        <CardHeader>
          <CardTitle>Generated Result</CardTitle>
          <CardDescription>
            Generated vector and metadata
          </CardDescription>
        </CardHeader>
        <CardContent>
          {result ? (
            <div className="space-y-4">
              {/* Result Summary */}
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="font-medium text-gray-700">Vector Length</p>
                  <p>{result.vector.length}</p>
                </div>
                <div>
                  <p className="font-medium text-gray-700">Expected Trend</p>
                  <p>{typeof result.expected_trend === 'number' 
                    ? result.expected_trend.toFixed(3) 
                    : result.expected_trend}</p>
                </div>
                <div>
                  <p className="font-medium text-gray-700">Algorithm Used</p>
                  <Badge variant="outline">{result.algorithm_used}</Badge>
                </div>
                <div>
                  <p className="font-medium text-gray-700">Generated At</p>
                  <p>{new Date(result.timestamp).toLocaleTimeString()}</p>
                </div>
              </div>

              {/* Vector Preview */}
              <div>
                <Label className="text-sm font-medium">Vector Preview (first 10 values)</Label>
                <Textarea
                  value={result.vector.slice(0, 10).map(v => v.toFixed(4)).join(', ') + '...'}
                  readOnly
                  className="mt-1 font-mono text-xs"
                  rows={3}
                />
              </div>

              {/* Action Buttons */}
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={copyVector}>
                  <Copy className="h-4 w-4 mr-2" />
                  Copy Vector
                </Button>
                <Button variant="outline" size="sm" onClick={downloadVector}>
                  <Download className="h-4 w-4 mr-2" />
                  Download JSON
                </Button>
              </div>

              {/* Vector Statistics */}
              <div className="bg-gray-50 p-3 rounded text-xs">
                <p><strong>Mean:</strong> {(result.vector.reduce((a, b) => a + b, 0) / result.vector.length).toFixed(4)}</p>
                <p><strong>Min:</strong> {Math.min(...result.vector).toFixed(4)}</p>
                <p><strong>Max:</strong> {Math.max(...result.vector).toFixed(4)}</p>
                <p><strong>Std Dev:</strong> {Math.sqrt(result.vector.reduce((acc, val) => {
                  const mean = result.vector.reduce((a, b) => a + b, 0) / result.vector.length;
                  return acc + Math.pow(val - mean, 2);
                }, 0) / result.vector.length).toFixed(4)}</p>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <Wand2 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">Generate a vector to see results</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default VectorGenerator;