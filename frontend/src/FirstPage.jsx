import React, { useRef } from 'react';

function FirstPage({ onUpload, onRecord, onMatch, isRecording, isAnalyzing, progress, hasAudio }) {
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    if (e.target.files.length > 0) {
      onUpload(e.target.files[0]);
    }
  };

  return (
    <div className="w-[375px] h-[812px] relative bg-gradient-to-br from-purple-50 to-blue-50 overflow-hidden">
      {/* Header Card */}
      <div className="w-[343px] h-[136px] absolute left-[16px] top-[42px] bg-white rounded-2xl shadow-sm">
        <div className="absolute left-[24px] top-[30px]">
          <div className="flex items-center gap-2">
            <h1 className="text-4xl font-bold text-gray-800">VibeMatch</h1>
            <div className="bg-violet-800/10 rounded-full px-3 py-2">
              <span className="text-sm font-medium text-indigo-950">+100M Songs</span>
            </div>
          </div>
          <div className="flex items-center gap-2 mt-4">
            <div className="flex gap-1">
              <div className="w-2 h-2 bg-cyan-400 rounded-full pulse"></div>
              <div className="w-2 h-2 bg-violet-500 rounded-full pulse" style={{ animationDelay: '0.3s' }}></div>
              <div className="w-2 h-2 bg-indigo-600 rounded-full pulse" style={{ animationDelay: '0.6s' }}></div>
            </div>
            <span className="text-sm text-gray-500">Find the music to your groove</span>
          </div>
        </div>
      </div>

      {/* Animated Sliders */}
      <div className="absolute left-[22px] top-[192px] w-[320px] h-[279px]">
        <svg width="320" height="279" viewBox="0 0 320 279" fill="none" className="-rotate-90">
          {[0, 1, 2, 3, 4].map((i) => (
            <g key={i} transform={`translate(30, ${i * 50})`}>
              <rect width="280" height="25" rx="15" fill={i % 2 === 0 ? "url(#grad1)" : "url(#grad2)"} stroke="black" strokeWidth="2"/>
              <circle cx={i * 50 + 100} cy="12.5" r="11.5" fill="#ECF0F3" stroke="black" strokeWidth="2">
                <animate attributeName="cx" from={i * 50 + 100} to={i * 50 + 150} dur="3s" repeatCount="indefinite"/>
              </circle>
            </g>
          ))}
          <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ECF0F3" />
              <stop offset="100%" stopColor="#1A074D" />
            </linearGradient>
            <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#023F4B" />
              <stop offset="100%" stopColor="#ECF0F3" />
            </linearGradient>
          </defs>
        </svg>
      </div>

      {/* Upload Button */}
      <button 
        onClick={() => fileInputRef.current?.click()}
        className={`absolute w-[106px] h-[84px] left-[62px] top-[476px] bg-white rounded-xl shadow-sm transition-all ${hasAudio ? 'bg-violet-50' : ''}`}
      >
        <div className="w-11 h-11 mx-auto mt-5 bg-violet-500/10 rounded-full flex items-center justify-center">
          <svg width="12" height="16" viewBox="0 0 12 16" fill="none">
            <path d="M6 1V11M6 1L2 5M6 1L10 5M1 15H11" stroke="#3E0BF5" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <span className="block mt-2 text-sm font-medium text-gray-700">Upload</span>
      </button>
      <input 
        ref={fileInputRef}
        type="file" 
        accept="audio/*" 
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />

      {/* Record Button */}
      <button 
        onClick={onRecord}
        className={`absolute w-[106px] h-[84px] left-[196px] top-[476px] bg-white rounded-xl shadow-sm transition-all ${isRecording ? 'bg-emerald-100' : ''}`}
      >
        <div className={`w-11 h-11 mx-auto mt-5 bg-emerald-500/10 rounded-full flex items-center justify-center ${isRecording ? 'pulse' : ''}`}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="7" stroke="#10B981" strokeWidth="2"/>
            <circle cx="8" cy="8" r="3" fill="#10B981"/>
          </svg>
        </div>
        <span className="block mt-2 text-sm font-medium text-gray-700">
          {isRecording ? 'Recording...' : 'Record'}
        </span>
      </button>

      {/* Hit the Match Button */}
      <button 
        onClick={onMatch}
        disabled={!hasAudio || isAnalyzing}
        className={`absolute w-[140px] h-[43px] left-[112px] top-[586px] rounded-xl shadow-sm transition-all ${
          hasAudio && !isAnalyzing ? 'bg-violet-950 hover:bg-violet-900' : 'bg-gray-400'
        }`}
      >
        <span className="text-lg font-semibold text-white">
          {isAnalyzing ? 'Analyzing...' : 'Hit the match'}
        </span>
      </button>

      {/* Progress Card */}
      {isAnalyzing && (
        <div className="absolute w-[339px] h-[56px] left-[18px] top-[656px] bg-white rounded-2xl px-4 py-3">
          <p className="text-sm text-gray-500 mb-2">Analysing Your Taste</p>
          <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-violet-500 to-amber-500 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default FirstPage;
