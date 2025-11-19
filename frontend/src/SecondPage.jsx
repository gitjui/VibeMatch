import React from 'react';

function SecondPage({ results, onBack }) {
  const exactMatch = results?.exact_match;
  const recommendations = results?.recommendations || [];

  return (
    <div className="w-[375px] h-[812px] bg-gradient-to-br from-purple-50 to-blue-50 overflow-y-auto">
      {/* Header */}
      <div className="sticky top-0 bg-white/80 backdrop-blur-sm px-4 py-4 flex items-center gap-3 border-b border-gray-200">
        <button onClick={onBack} className="text-2xl">←</button>
        <h1 className="text-xl font-bold text-gray-800">Your Match</h1>
      </div>

      {/* Match Result */}
      {exactMatch?.found ? (
        <div className="p-4">
          {/* Match Badge */}
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center gap-2 bg-emerald-500 text-white px-4 py-2 rounded-full">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M13.33 4L6 11.33L2.67 8" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <span className="text-sm font-medium">Match Found</span>
            </div>
            <div className="bg-violet-500/10 text-violet-600 px-4 py-2 rounded-full font-bold">
              {Math.round(exactMatch.confidence * 100)}%
            </div>
          </div>

          {/* Song Card */}
          <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
            <div className="flex flex-col items-center">
              {/* Album Art */}
              <div className="w-32 h-32 bg-gradient-to-br from-violet-500 to-pink-500 rounded-2xl flex items-center justify-center text-6xl text-white mb-4">
                ♪
              </div>
              
              {/* Song Info */}
              <h2 className="text-2xl font-bold text-gray-800 text-center mb-2">
                {exactMatch.title || 'Unknown Title'}
              </h2>
              <p className="text-lg text-gray-600 mb-1">
                {exactMatch.artist || 'Unknown Artist'}
              </p>
              <p className="text-sm text-gray-400">
                {results.query?.file || 'Audio File'}
              </p>
            </div>
          </div>

          {/* Recommendations */}
          {recommendations.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-3">Similar Songs</h3>
              <div className="space-y-3">
                {recommendations.map((rec, index) => (
                  <div key={index} className="bg-white rounded-xl p-4 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex justify-between items-center">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-medium text-violet-600 bg-violet-50 px-2 py-1 rounded">
                            #{index + 1}
                          </span>
                          <p className="font-medium text-gray-800">{rec.title || 'Unknown'}</p>
                        </div>
                        <p className="text-sm text-gray-500">{rec.artist || 'Unknown Artist'}</p>
                        {rec.genre && (
                          <span className="text-xs text-gray-400 mt-1 inline-block">
                            {rec.genre} • {rec.year || 'N/A'}
                          </span>
                        )}
                      </div>
                      <div className="text-lg font-semibold text-violet-600">
                        {Math.round(rec.similarity * 100)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="p-6 text-center">
          <div className="text-6xl mb-4">🎵</div>
          <h2 className="text-xl font-bold text-gray-800 mb-2">No Match Found</h2>
          <p className="text-gray-600 mb-6">Try recording a different part of the song or upload another file.</p>
          <button 
            onClick={onBack}
            className="bg-violet-600 text-white px-6 py-3 rounded-xl font-semibold hover:bg-violet-700"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
}

export default SecondPage;
