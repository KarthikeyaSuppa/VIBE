import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Star, RefreshCw } from 'lucide-react';

interface StoryDialogProps {
  onClose: () => void;
}

const API_URL = process.env.NODE_ENV === 'production'
  ? 'https://vibe-yxg8.onrender.com'
  : 'http://localhost:8000';

const formatStory = (rawStory: string): string => {
  // Remove the initial greeting if present
  const storyWithoutGreeting = rawStory.replace(
    "I'd be delighted to spin a yarn for you! Here's the story:",
    ''
  ).trim();

  // Split into paragraphs and format
  const paragraphs = storyWithoutGreeting.split('\n\n');
  return paragraphs.join('\n\n');
};

const StoryDialog: React.FC<StoryDialogProps> = ({ onClose }) => {
  const [step, setStep] = useState<'welcome' | 'prompt' | 'story' | 'feedback' | 'another'>('welcome');
  const [prompt, setPrompt] = useState('');
  const [story, setStory] = useState('');
  const [displayedStory, setDisplayedStory] = useState('');
  const [rating, setRating] = useState(0);
  const [feedback, setFeedback] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (story && step === 'story') {
      setIsTyping(true);
      const formattedStory = formatStory(story);
      let index = 0;
      
      const interval = setInterval(() => {
        if (index <= formattedStory.length) {
          setDisplayedStory(prev => formattedStory.slice(0, index));
          index++;
        } else {
          setIsTyping(false);
          clearInterval(interval);
        }
      }, 5);
      
      return () => {
        if (step !== 'story') {
          clearInterval(interval);
        }
      };
    }
  }, [story, step]);

  const handleSubmitPrompt = async () => {
    if (!prompt.trim()) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/generate-story`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ query: prompt }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate story');
      }

      const data = await response.json();
      setStory(data.story);
      setStep('story');
    } catch (err) {
      setError(err.message || 'Failed to generate story. Please try again.');
      console.error('Error generating story:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmitFeedback = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/store-feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query_text: prompt,
          story_text: story,
          rating,
          feedback_text: feedback || null,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to submit feedback');
      }

      setStep('another');
    } catch (err) {
      setError('Failed to submit feedback. Please try again.');
      console.error('Error submitting feedback:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStartOver = () => {
    setPrompt('');
    setStory('');
    setDisplayedStory('');
    setRating(0);
    setFeedback('');
    setError(null);
    setStep('prompt');
  };

  const renderStory = () => {
    return (
      <>
        <p className="text-purple-400 mb-6 text-lg italic">
          I'd be delighted to spin a yarn for you! Here's the story:
        </p>
        {displayedStory.split('\n\n').map((paragraph, index) => (
          <p key={index} className="text-gray-200 mb-6 leading-relaxed text-lg">
            {paragraph}
          </p>
        ))}
      </>
    );
  };

  const renderStep = () => {
    switch (step) {
      case 'welcome':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            <h2 className="text-2xl font-bold mb-4">Welcome to the Enhanced Story Generator!</h2>
            <p className="text-gray-300 mb-6">You can request stories, and I'll create them using character relationships when available.</p>
            <button
              onClick={() => setStep('prompt')}
              className="px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg flex items-center justify-center space-x-2 mx-auto"
            >
              <Sparkles className="w-5 h-5" />
              <span>Begin Your Story</span>
            </button>
          </motion.div>
        );

      case 'prompt':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-2xl mx-auto"
          >
            <h2 className="text-2xl font-bold mb-4">What story would you like me to create?</h2>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full h-32 bg-gray-800 text-white rounded-lg p-4 mb-4 focus:ring-2 focus:ring-purple-500 focus:outline-none"
              placeholder="What kind of story would you like me to create?"
              disabled={isLoading}
            />
            {error && (
              <p className="text-red-500 mb-4">{error}</p>
            )}
            <button
              onClick={handleSubmitPrompt}
              disabled={isLoading || !prompt.trim()}
              className={`px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg flex items-center justify-center space-x-2 ${
                isLoading || !prompt.trim() ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {isLoading ? (
                <RefreshCw className="w-5 h-5 animate-spin" />
              ) : (
                <Sparkles className="w-5 h-5" />
              )}
              <span>{isLoading ? 'Generating...' : 'Generate Story'}</span>
            </button>
          </motion.div>
        );

      case 'story':
        return (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-3xl mx-auto"
          >
            <h2 className="text-2xl font-bold mb-4">Your Story</h2>
            <div className="bg-gray-800 rounded-lg p-6 mb-4 text-white">
              {renderStory()}
              {isTyping && <span className="animate-pulse">|</span>}
            </div>
            {!isTyping && (
              <button
                onClick={() => setStep('feedback')}
                className="px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg"
              >
                Provide Feedback
              </button>
            )}
          </motion.div>
        );

      case 'feedback':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-2xl mx-auto"
          >
            <h2 className="text-2xl font-bold mb-4">How was the story?</h2>
            <div className="flex justify-center space-x-2 mb-6">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  onClick={() => setRating(star)}
                  disabled={isLoading}
                  className={`transform hover:scale-110 transition-transform ${
                    star <= rating ? 'text-yellow-400' : 'text-gray-600'
                  }`}
                >
                  <Star className="w-8 h-8" fill={star <= rating ? 'currentColor' : 'none'} />
                </button>
              ))}
            </div>
            <textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              disabled={isLoading}
              className="w-full h-32 bg-gray-800 text-white rounded-lg p-4 mb-4 focus:ring-2 focus:ring-purple-500 focus:outline-none"
              placeholder="Any additional feedback about the characters or story? (Optional)"
            />
            {error && (
              <p className="text-red-500 mb-4">{error}</p>
            )}
            <button
              onClick={handleSubmitFeedback}
              disabled={isLoading || rating === 0}
              className={`px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg flex items-center justify-center space-x-2 ${
                isLoading || rating === 0 ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {isLoading ? (
                <RefreshCw className="w-5 h-5 animate-spin" />
              ) : (
                <span>Submit Feedback</span>
              )}
            </button>
          </motion.div>
        );

      case 'another':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            <h2 className="text-2xl font-bold mb-4">Thank you for your feedback!</h2>
            <p className="text-gray-300 mb-6">It helps improve future stories.</p>
            <p className="text-xl mb-6">Would you like to generate another story?</p>
            <div className="flex justify-center space-x-4">
              <button
                onClick={handleStartOver}
                className="px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg flex items-center space-x-2"
              >
                <RefreshCw className="w-5 h-5" />
                <span>Create Another Story</span>
              </button>
              <button
                onClick={onClose}
                className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg"
              >
                Close
              </button>
            </div>
          </motion.div>
        );
    }
  };

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-8">
      {renderStep()}
    </div>
  );
};

export default StoryDialog;
