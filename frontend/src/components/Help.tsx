import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Database, Cpu, Search, MessageSquare, Star, RefreshCw } from 'lucide-react';

const Help = () => {
  const [activeTooltip, setActiveTooltip] = useState<number | null>(null);

  const steps = [
    {
      icon: <Database className="w-8 h-8" />,
      title: "1. Database Creation",
      description: "Data for characters, scenes, and stories is collected and stored with unique identifiers.",
      color: "text-green-500",
      details: {
        title: "Database Structure and Collections",
        content: [
          {
            subtitle: "Collections",
            items: [
              "Characters: Stores character data with traits, backstories, and relationships",
              "Scenes: Contains scene descriptions, locations, and participating characters",
              "Stories: Holds complete stories with metadata and plot points"
            ]
          },
          {
            subtitle: "Data Storage",
            items: [
              "MongoDB for efficient document storage and retrieval",
              "Unique identifiers (_id) for each document",
              "Metadata for enhanced searchability"
            ]
          },
          {
            subtitle: "Implementation",
            items: [
              "PyMongo client for database operations",
              "Automatic indexing for optimized queries",
              "Relationship mapping between collections"
            ]
          }
        ]
      }
    },
    {
      icon: <Cpu className="w-8 h-8" />,
      title: "2. Embedding Generation",
      description: "Text data is processed using ColBERT-XM model to generate meaningful embeddings.",
      color: "text-blue-500",
      details: {
        title: "Embedding Generation Process",
        content: [
          {
            subtitle: "Text Processing",
            items: [
              "Tokenization using ColBERT-XM tokenizer",
              "Maximum sequence length of 512 tokens",
              "Padding and truncation handling"
            ]
          },
          {
            subtitle: "Model Architecture",
            items: [
              "ColBERT-XM for contextual embeddings",
              "Mean pooling of last hidden states",
              "Dimensionality reduction for efficiency"
            ]
          },
          {
            subtitle: "Output Format",
            items: [
              "Dense vector representations",
              "Metadata preservation",
              "Collection-specific embedding strategies"
            ]
          }
        ]
      }
    },
    {
      icon: <Search className="w-8 h-8" />,
      title: "3. Vector Search",
      description: "User queries are matched against stored vectors to find relevant content.",
      color: "text-purple-500",
      details: {
        title: "Vector Search System",
        content: [
          {
            subtitle: "Search Process",
            items: [
              "Query embedding generation",
              "Pinecone vector similarity search",
              "Top-k retrieval with metadata"
            ]
          },
          {
            subtitle: "Ranking",
            items: [
              "Cosine similarity scoring",
              "Feedback-based result adjustment",
              "Dynamic result filtering"
            ]
          },
          {
            subtitle: "Optimization",
            items: [
              "Caching for frequent queries",
              "Batch processing for efficiency",
              "Result diversity enforcement"
            ]
          }
        ]
      }
    },
    {
      icon: <MessageSquare className="w-8 h-8" />,
      title: "4. Story Generation",
      description: "Llama 3.3-70B generates engaging stories using matched content and relationships.",
      color: "text-indigo-500",
      details: {
        title: "Story Generation Pipeline",
        content: [
          {
            subtitle: "Content Preparation",
            items: [
              "Context assembly from search results",
              "Character relationship analysis",
              "Genre and theme detection"
            ]
          },
          {
            subtitle: "Generation Process",
            items: [
              "Llama 3.3-70B model integration",
              "Prompt engineering for coherence",
              "Multi-step generation pipeline"
            ]
          },
          {
            subtitle: "Enhancement",
            items: [
              "Character arc development",
              "Plot consistency checking",
              "Narrative flow optimization"
            ]
          }
        ]
      }
    },
    {
      icon: <Star className="w-8 h-8" />,
      title: "5. Feedback System",
      description: "User feedback helps improve future story generation and recommendations.",
      color: "text-yellow-500",
      details: {
        title: "Feedback Collection and Processing",
        content: [
          {
            subtitle: "Data Collection",
            items: [
              "User ratings (1-5 stars)",
              "Textual feedback processing",
              "Interaction metrics tracking"
            ]
          },
          {
            subtitle: "Analysis",
            items: [
              "Sentiment analysis of feedback",
              "Pattern recognition in ratings",
              "Content improvement suggestions"
            ]
          },
          {
            subtitle: "Implementation",
            items: [
              "Feedback storage in MongoDB",
              "Rating-based result adjustment",
              "Continuous system improvement"
            ]
          }
        ]
      }
    },
    {
      icon: <RefreshCw className="w-8 h-8" />,
      title: "6. Enhanced Generation",
      description: "Stories are refined using graph-based relationship analysis for emotional depth.",
      color: "text-pink-500",
      details: {
        title: "Story Enhancement Process",
        content: [
          {
            subtitle: "Relationship Analysis",
            items: [
              "Graph-based character mapping",
              "Relationship path analysis",
              "Emotional context extraction"
            ]
          },
          {
            subtitle: "Content Refinement",
            items: [
              "Character interaction enhancement",
              "Plot coherence verification",
              "Narrative depth addition"
            ]
          },
          {
            subtitle: "Quality Assurance",
            items: [
              "Consistency checking",
              "Style and tone verification",
              "Final polish application"
            ]
          }
        ]
      }
    }
  ];

  return (
    <div className="min-h-screen bg-gray-900 py-24">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-6">How VIBE Works</h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Our advanced AI system combines multiple technologies to create engaging, personalized stories.
            Here's how the magic happens:
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="relative group"
              onMouseEnter={() => setActiveTooltip(index)}
              onMouseLeave={() => setActiveTooltip(null)}
            >
              <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 hover:transform hover:scale-105 transition-all cursor-pointer">
                <div className={`${step.color} mb-4`}>{step.icon}</div>
                <h3 className="text-xl font-semibold text-white mb-3">{step.title}</h3>
                <p className="text-gray-300">{step.description}</p>
              </div>

              {activeTooltip === index && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="absolute z-50 w-96 bg-gray-800 rounded-xl p-6 shadow-xl left-full ml-4 top-0"
                >
                  <h4 className="text-lg font-semibold text-white mb-4">{step.details.title}</h4>
                  {step.details.content.map((section, sIndex) => (
                    <div key={sIndex} className="mb-4">
                      <h5 className="text-sm font-medium text-gray-300 mb-2">{section.subtitle}</h5>
                      <ul className="list-disc list-inside text-sm text-gray-400">
                        {section.items.map((item, iIndex) => (
                          <li key={iIndex} className="mb-1">{item}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </motion.div>
              )}
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-16 text-center"
        >
          <h2 className="text-3xl font-bold text-white mb-4">Ready to Create Your Story?</h2>
          <p className="text-gray-300 mb-8">
            Experience the power of AI-driven storytelling and create your own magical tales.
          </p>
          <a
            href="/"
            className="inline-flex items-center space-x-2 px-8 py-4 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transform hover:scale-105 transition-all"
          >
            <span>Start Creating</span>
          </a>
        </motion.div>
      </div>
    </div>
  );
};

export default Help;