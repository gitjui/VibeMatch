import React, { useState } from 'react';
import './App.css';
import FirstPage from './FirstPage';
import SecondPage from './SecondPage';

function App() {
  const [showResults, setShowResults] = useState(false);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleIdentify = async (audioFile) => {
    setIsLoading(true);
    
    const formData = new FormData();
    formData.append('file', audioFile);

    try {
      const response = await fetch('http://localhost:8000/identify', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
      setShowResults(true);
    } catch (error) {
      console.error('Error identifying song:', error);
      alert('Error connecting to API. Make sure the server is running at http://localhost:8000');
    } finally {
      setIsLoading(false);
    }
  };

  const handleBack = () => {
    setShowResults(false);
    setResults(null);
  };

  return (
    <div className="App">
      {!showResults ? (
        <FirstPage onIdentify={handleIdentify} isLoading={isLoading} />
      ) : (
        <SecondPage results={results} onBack={handleBack} />
      )}
    </div>
  );
}

export default App;
