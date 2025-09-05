import React from "react";
import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import TrendDetectorDashboard from "./components/TrendDetectorDashboard";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<TrendDetectorDashboard />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;