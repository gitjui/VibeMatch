import React, { useState } from 'react';
import FirstPage from './FirstPage.jsx';
import SecondPage from './SecondPage.jsx';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [page, setPage] = useState('first'); // 'first' or 'second'
  const [audioFile, setAudioFile] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);

  // Handle Upload
  const handleUpload = (file) => {
    setAudioFile(file);
    console.log('File selected:', file.name);
  };

  // Handle Record
  const handleRecord = async () => {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        const audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
          audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
          const file = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
          setAudioFile(file);
          setIsRecording(false);
          stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        setIsRecording(true);

        // Auto stop after 5 seconds
        setTimeout(() => {
          if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
          }
        }, 5000);
      } catch (error) {
        console.error('Microphone error:', error);
        alert('Cannot access microphone. Please check permissions.');
      }
    }
  };

  // Hit the Match - Call API
  const handleMatch = async () => {
    if (!audioFile) {
      alert('Please upload or record audio first!');
      return;
    }

    setIsAnalyzing(true);
    setProgress(0);

    // Animate progress
    const interval = setInterval(() => {
      setProgress(prev => Math.min(prev + 10, 90));
    }, 200);

    try {
      const formData = new FormData();
      formData.append('file', audioFile);

      const response = await fetch(`${API_BASE_URL}/identify`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error('API error');

      const data = await response.json();
      clearInterval(interval);
      setProgress(100);
      
      setTimeout(() => {
        setResults(data);
        setPage('second');
        setIsAnalyzing(false);
      }, 500);

    } catch (error) {
      clearInterval(interval);
      setIsAnalyzing(false);
      console.error('Error:', error);
      alert('Error identifying song. Make sure API is running at ' + API_BASE_URL);
    }
  };

  const handleBack = () => {
    setPage('first');
    setAudioFile(null);
    setResults(null);
    setProgress(0);
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-100">
      {page === 'first' ? (
        <FirstPage 
          onUpload={handleUpload}
          onRecord={handleRecord}
          onMatch={handleMatch}
          isRecording={isRecording}
          isAnalyzing={isAnalyzing}
          progress={progress}
          hasAudio={!!audioFile}
        />
      ) : (
        <SecondPage 
          results={results}
          onBack={handleBack}
        />
      )}
    </div>
  );
}

export default App;
