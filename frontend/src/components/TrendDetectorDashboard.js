import React, { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import ModelDashboard from './ModelDashboard';
import DriftTester from './DriftTester';
import VectorGenerator from './VectorGenerator';
import PredictionTester from './PredictionTester';
import { apiService } from '../services/apiService';
import { useToast } from "../hooks/use-toast";
import { Toaster } from "./ui/toaster";

const TrendDetectorDashboard = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const modelsData = await apiService.getModels();
      setModels(modelsData);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch model information",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const refreshModels = () => {
    setLoading(true);
    fetchModels();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading TrendDetector...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">TrendDetector</h1>
              <p className="text-gray-600 mt-2">Dual-Model Content Trend Detection System</p>
            </div>
            <div className="flex gap-2">
              <Badge variant="outline" className="px-3 py-1">
                Models Loaded: {models.filter(m => m.is_loaded).length}/{models.length}
              </Badge>
              <Button variant="outline" size="sm" onClick={refreshModels}>
                Refresh
              </Button>
            </div>
          </div>
        </div>

        {/* Main Tabs */}
        <Tabs defaultValue="models" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="models">Models</TabsTrigger>
            <TabsTrigger value="drift-test">Drift Testing</TabsTrigger>
            <TabsTrigger value="vector-gen">Vector Generator</TabsTrigger>
            <TabsTrigger value="prediction">Predictions</TabsTrigger>
          </TabsList>

          <TabsContent value="models" className="space-y-6">
            <ModelDashboard models={models} onRefresh={refreshModels} />
          </TabsContent>

          <TabsContent value="drift-test" className="space-y-6">
            <DriftTester models={models} />
          </TabsContent>

          <TabsContent value="vector-gen" className="space-y-6">
            <VectorGenerator />
          </TabsContent>

          <TabsContent value="prediction" className="space-y-6">
            <PredictionTester models={models} />
          </TabsContent>
        </Tabs>
      </div>
      <Toaster />
    </div>
  );
};

export default TrendDetectorDashboard;