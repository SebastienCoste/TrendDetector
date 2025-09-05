import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { CheckCircle, AlertCircle, Clock, Activity } from "lucide-react";
import { apiService } from '../services/apiService';

const ModelDashboard = ({ models, onRefresh }) => {
  const [healthData, setHealthData] = useState(null);
  const [loading, setLoading] = useState(false);

  const checkHealth = async () => {
    setLoading(true);
    try {
      // Use the backend health endpoint
      const backendUrl = process.env.REACT_APP_BACKEND_URL;
      const response = await fetch(`${backendUrl}/health`);
      const healthData = await response.json();
      setHealthData(healthData);
    } catch (error) {
      console.error('Health check failed:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  const getModelStatusIcon = (model) => {
    if (model.is_loaded) {
      return <CheckCircle className="h-5 w-5 text-green-500" />;
    } else {
      return <AlertCircle className="h-5 w-5 text-yellow-500" />;
    }
  };

  const getModelStatusBadge = (model) => {
    if (model.is_loaded) {
      return <Badge variant="default" className="bg-green-100 text-green-800">Loaded</Badge>;
    } else {
      return <Badge variant="secondary">Not Loaded</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      {/* System Health Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                System Health
              </CardTitle>
              <CardDescription>TrendDetector service status and health metrics</CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={checkHealth} disabled={loading}>
              {loading ? "Checking..." : "Check Health"}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {healthData ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="flex items-center justify-center mb-2">
                  {healthData.status === 'healthy' ? (
                    <CheckCircle className="h-8 w-8 text-green-500" />
                  ) : (
                    <AlertCircle className="h-8 w-8 text-red-500" />
                  )}
                </div>
                <p className="text-sm font-medium">Service Status</p>
                <p className="text-xs text-gray-600 capitalize">{healthData.status}</p>
              </div>
              <div className="text-center">
                <div className="flex items-center justify-center mb-2">
                  {healthData.gpu_enabled ? (
                    <CheckCircle className="h-8 w-8 text-blue-500" />
                  ) : (
                    <Clock className="h-8 w-8 text-gray-400" />
                  )}
                </div>
                <p className="text-sm font-medium">GPU Status</p>
                <p className="text-xs text-gray-600">{healthData.gpu_enabled ? 'Enabled' : 'CPU Only'}</p>
              </div>
              <div className="text-center">
                <div className="flex items-center justify-center mb-2">
                  <Badge variant={healthData.model_loaded ? "default" : "secondary"}>
                    {healthData.model_loaded ? 'Loaded' : 'No Model'}
                  </Badge>
                </div>
                <p className="text-sm font-medium">Model Status</p>
                <p className="text-xs text-gray-600">Version {healthData.version}</p>
              </div>
            </div>
          ) : (
            <div className="text-center py-4">
              <p className="text-gray-500">Click "Check Health" to view system status</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Models Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {models.map((model) => (
          <Card key={model.model_name}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getModelStatusIcon(model)}
                  <CardTitle className="text-lg">{model.model_name}</CardTitle>
                </div>
                {getModelStatusBadge(model)}
              </div>
              <CardDescription>
                {model.model_type === 'classification' 
                  ? 'Predicts categorical trends (upward/downward/neutral)'
                  : 'Predicts continuous trend scores (-1 to 1)'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="font-medium text-gray-700">Model Type</p>
                    <p className="capitalize">{model.model_type}</p>
                  </div>
                  <div>
                    <p className="font-medium text-gray-700">Version</p>
                    <p>{model.version}</p>
                  </div>
                </div>
                
                {model.last_updated && (
                  <div>
                    <p className="font-medium text-gray-700 text-sm">Last Updated</p>
                    <p className="text-sm text-gray-600">
                      {new Date(model.last_updated).toLocaleString()}
                    </p>
                  </div>
                )}

                <div className="pt-2">
                  <div className="flex justify-between text-sm mb-1">
                    <span>Model Readiness</span>
                    <span>{model.is_loaded ? '100%' : '0%'}</span>
                  </div>
                  <Progress value={model.is_loaded ? 100 : 0} className="h-2" />
                </div>

                {model.model_type === 'classification' && (
                  <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
                    <strong>Output:</strong> upward, downward, neutral + confidence + probabilities
                  </div>
                )}
                
                {model.model_type === 'regression' && (
                  <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
                    <strong>Output:</strong> continuous score [-1, 1] + confidence
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {models.length === 0 && (
        <Card>
          <CardContent className="text-center py-12">
            <AlertCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500 mb-2">No models available</p>
            <Button variant="outline" onClick={onRefresh}>
              Refresh Models
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ModelDashboard;