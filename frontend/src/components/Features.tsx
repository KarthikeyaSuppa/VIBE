import React from 'react';
import { Brain, Wand2, Users } from 'lucide-react';

const Features = () => {
  const features = [
    {
      icon: <Brain className="w-8 h-8" />,
      title: "AI-Powered Storytelling",
      description: "Advanced algorithms create unique, engaging narratives tailored to your preferences."
    },
    {
      icon: <Wand2 className="w-8 h-8" />,
      title: "Magical Experiences",
      description: "Immerse yourself in enchanting worlds with rich characters and compelling plots."
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: "Community Driven",
      description: "Share your stories, collaborate with others, and explore a universe of creativity."
    }
  ];

  return (
    <section className="py-24 bg-gray-900">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">Experience the Magic</h2>
          <p className="text-gray-400">Discover the power of AI-driven storytelling</p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-12">
          {features.map((feature, index) => (
            <div key={index} className="bg-gray-800/50 rounded-xl p-8 backdrop-blur-sm hover:transform hover:scale-105 transition-all">
              <div className="text-purple-500 mb-4">{feature.icon}</div>
              <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
              <p className="text-gray-400">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;