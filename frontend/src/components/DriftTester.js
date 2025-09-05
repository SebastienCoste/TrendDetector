import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Label } from "./ui/label";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { 
  Play, Square, BarChart3, TrendingUp, AlertTriangle, 
  Clock, Target, Activity, Trash2 
} from "lucide-react";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, ScatterPlot, Scatter, ReferenceLine
} from 'recharts';
import { apiService } from '../services/apiService';
import { useToast } from "../hooks/use-toast";

const DriftTester = ({ models }) => {
  const [activeTests, setActiveTests] = useState([]);
  const [selectedTest, setSelectedTest] = useState(null);
  const [testConfig, setTestConfig] = useState({
    num_requests: 100,
    feedback_frequency: 10,
    drift_point: 50,
    model_type: 'regression'
  });
  const [testStatus, setTestStatus] = useState(null);
  const [testResults, setTestResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const intervalRef = useRef(null);
  const { toast } = useToast();

  useEffect(() => {
    fetchActiveTests();
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (selectedTest) {
      startMonitoring(selectedTest);
    } else {
      stopMonitoring();
    }
  }, [selectedTest]);

  const fetchActiveTests = async () => {
    try {
      const tests = await apiService.getActiveTests();
      setActiveTests(tests);
    } catch (error) {
      console.error('Failed to fetch active tests:', error);
    }
  };

  const startDriftTest = async () => {
    setLoading(true);
    try {
      const result = await apiService.startDriftTest(testConfig);
      setSelectedTest(result.test_id);
      await fetchActiveTests();
      
      toast({
        title: "Test Started",
        description: `Drift test ${result.test_id.substring(0, 8)}... started`,
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to start drift test",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const startMonitoring = (testId) => {
    stopMonitoring();
    
    const fetchData = async () => {
      try {
        const [status, results] = await Promise.all([
          apiService.getDriftTestStatus(testId),
          apiService.getDriftTestResults(testId, 100) // Get latest 100 results
        ]);
        
        setTestStatus(status);
        setTestResults(results);
        
        if (status.status === 'completed' || status.status === 'error') {
          stopMonitoring();
        }
      } catch (error) {
        console.error('Failed to fetch test data:', error);
        if (error.response?.status === 404) {
          setSelectedTest(null);
          setTestStatus(null);
          setTestResults([]);
          fetchActiveTests();
        }
      }
    };

    // Initial fetch
    fetchData();
    
    // Set up polling
    intervalRef.current = setInterval(fetchData, 2000);
  };

  const stopMonitoring = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const cleanupTest = async (testId) => {
    try {
      await apiService.cleanupDriftTest(testId);
      if (selectedTest === testId) {
        setSelectedTest(null);
        setTestStatus(null);
        setTestResults([]);
      }
      await fetchActiveTests();
      
      toast({
        title: "Test Cleaned Up",
        description: "Test data has been removed",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to cleanup test",
        variant: "destructive"
      });
    }
  };

  const selectTest = (testId) => {
    setSelectedTest(testId);
  };

  // Prepare chart data
  const chartData = testResults.map((result, index) => ({
    request: result.request_id,
    error: result.absolute_error,
    confidence: result.confidence,
    isDrift: result.is_drift_period,
    feedbackProvided: result.feedback_provided,
    predicted: typeof result.predicted_trend === 'number' 
      ? result.predicted_trend 
      : (result.predicted_trend === 'upward' ? 1 : result.predicted_trend === 'downward' ? -1 : 0),
    expected: typeof result.expected_trend === 'number' 
      ? result.expected_trend 
      : (result.expected_trend === 'upward' ? 1 : result.expected_trend === 'downward' ? -1 : 0)
  }));

  const availableModels = models.filter(m => m.is_loaded);

  return (
    <div className="space-y-6">
      {/* Test Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Drift Test Configuration
          </CardTitle>
          <CardDescription>
            Configure and start concept drift evaluation tests
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="space-y-2">
              <Label>Model Type</Label>
              <Select
                value={testConfig.model_type}
                onValueChange={(value) => setTestConfig(prev => ({ ...prev, model_type: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="classification">Classification</SelectItem>
                  <SelectItem value="regression">Regression</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label>Total Requests</Label>
              <Input
                type="number"
                value={testConfig.num_requests}
                onChange={(e) => setTestConfig(prev => ({ ...prev, num_requests: parseInt(e.target.value) }))}
                min={10}
                max={1000}
              />
            </div>
            
            <div className="space-y-2">
              <Label>Drift Point</Label>
              <Input
                type="number"
                value={testConfig.drift_point}
                onChange={(e) => setTestConfig(prev => ({ ...prev, drift_point: parseInt(e.target.value) }))}
                min={5}
                max={testConfig.num_requests - 5}
              />
            </div>
            
            <div className="space-y-2">
              <Label>Feedback Frequency</Label>
              <Input
                type="number"
                value={testConfig.feedback_frequency}
                onChange={(e) => setTestConfig(prev => ({ ...prev, feedback_frequency: parseInt(e.target.value) }))}
                min={1}
                max={50}
              />
            </div>
          </div>
          
          <Button 
            onClick={startDriftTest} 
            disabled={loading || availableModels.length === 0}
            className="w-full"
          >
            <Play className="h-4 w-4 mr-2" />
            {loading ? "Starting Test..." : "Start Drift Test"}
          </Button>
          
          {availableModels.length === 0 && (
            <p className="text-xs text-red-500 mt-2">No models are currently loaded</p>
          )}
        </CardContent>
      </Card>

      {/* Active Tests & Test Selection */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Active Tests</CardTitle>
            <CardDescription>Currently running or completed tests</CardDescription>
          </CardHeader>
          <CardContent>
            {activeTests.length > 0 ? (
              <div className="space-y-2">
                {activeTests.map(testId => (
                  <div 
                    key={testId}
                    className={`p-2 border rounded cursor-pointer transition-colors ${
                      selectedTest === testId ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => selectTest(testId)}
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-mono">{testId.substring(0, 8)}...</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          cleanupTest(testId);
                        }}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-sm">No active tests</p>
            )}
          </CardContent>
        </Card>

        {/* Test Status */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Test Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            {testStatus ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Badge variant={
                    testStatus.status === 'completed' ? 'default' : 
                    testStatus.status === 'error' ? 'destructive' : 
                    'secondary'
                  }>
                    {testStatus.status.toUpperCase()}
                  </Badge>
                  <span className="text-sm text-gray-600">
                    {testStatus.current_request} / {testStatus.total_requests}
                  </span>
                </div>
                
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Progress</span>
                    <span>{(testStatus.progress * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={testStatus.progress * 100} />
                </div>

                {testStatus.metrics && Object.keys(testStatus.metrics).length > 0 && (
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="font-medium">Feedback Events</p>
                      <p>{testStatus.metrics.feedback_events || 0}</p>
                    </div>
                    <div>
                      <p className="font-medium">Drift Detections</p>
                      <p>{testStatus.metrics.drift_detections || 0}</p>
                    </div>
                    {testStatus.metrics.error_change_percent && (
                      <div className="col-span-2">
                        <p className="font-medium">Error Change</p>
                        <p className={testStatus.metrics.error_change_percent > 0 ? 'text-red-600' : 'text-green-600'}>
                          {testStatus.metrics.error_change_percent.toFixed(1)}%
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8">
                <Clock className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                <p className="text-gray-500">Select a test to view status</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Visualization */}
      {testResults.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Drift Analysis Visualization
            </CardTitle>
            <CardDescription>
              Real-time visualization of concept drift test results
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="error" className="space-y-4">
              <TabsList>
                <TabsTrigger value="error">Prediction Error</TabsTrigger>
                <TabsTrigger value="comparison">True vs Predicted</TabsTrigger>
                <TabsTrigger value="confidence">Confidence</TabsTrigger>
              </TabsList>

              <TabsContent value="error" className="space-y-4">
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="request" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      {/* Drift introduction line */}
                      <ReferenceLine 
                        x={testConfig.drift_point} 
                        stroke="#ef4444" 
                        strokeDasharray="5 5"
                        label="Drift Introduced"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="error" 
                        stroke="#3b82f6" 
                        strokeWidth={2}
                        dot={false}
                        name="Prediction Error"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </TabsContent>

              <TabsContent value="comparison" className="space-y-4">
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="request" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <ReferenceLine 
                        x={testConfig.drift_point} 
                        stroke="#ef4444" 
                        strokeDasharray="5 5"
                        label="Drift Introduced"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="expected" 
                        stroke="#22c55e" 
                        strokeWidth={2}
                        dot={false}
                        name="Expected"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="predicted" 
                        stroke="#3b82f6" 
                        strokeWidth={2}
                        dot={false}
                        name="Predicted"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </TabsContent>

              <TabsContent value="confidence" className="space-y-4">
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="request" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip />
                      <Legend />
                      <ReferenceLine 
                        x={testConfig.drift_point} 
                        stroke="#ef4444" 
                        strokeDasharray="5 5"
                        label="Drift Introduced"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="confidence" 
                        stroke="#8b5cf6" 
                        strokeWidth={2}
                        dot={false}
                        name="Confidence"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </TabsContent>
            </Tabs>

            {/* Summary Stats */}
            {testStatus?.metrics && (
              <div className="mt-6 p-4 bg-gray-50 rounded">
                <h4 className="font-medium mb-2">Test Summary</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-gray-600">Pre-drift Error</p>
                    <p className="font-medium">
                      {testStatus.metrics.pre_drift_error?.mean?.toFixed(4) || 'N/A'}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600">Drift Error</p>
                    <p className="font-medium">
                      {testStatus.metrics.drift_error?.mean?.toFixed(4) || 'N/A'}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600">Error Change</p>
                    <p className={`font-medium ${
                      testStatus.metrics.error_change_percent > 0 ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {testStatus.metrics.error_change_percent?.toFixed(1) || 'N/A'}%
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600">Results Count</p>
                    <p className="font-medium">{testResults.length}</p>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default DriftTester;