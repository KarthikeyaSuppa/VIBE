import { motion } from 'framer-motion';
import { BookOpen, History, Cpu, Sparkles, Brain, Palette } from 'lucide-react';

const About = () => {
  const fadeIn = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 }
  };

  const sections = [
    {
      icon: <History className="w-8 h-8" />,
      title: "The Art of Storytelling",
      content: "Storytelling is humanity's oldest tradition - a bridge connecting our past, present, and future. It's how we preserve history, share experiences, and build relationships across generations.",
      color: "from-blue-500 to-purple-500"
    },
    {
      icon: <Cpu className="w-8 h-8" />,
      title: "AI-Powered Innovation",
      content: "By combining retrieval-augmented generation with state-of-the-art LLMs like Llama 3.3-70B, we're revolutionizing how stories are created and experienced.",
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: <Brain className="w-8 h-8" />,
      title: "Intelligent Architecture",
      content: "Our system leverages MongoDB for structured data, Pinecone for vector-based semantic search, and ColBERT-XM for deep understanding, creating contextually rich and personalized narratives.",
      color: "from-pink-500 to-red-500"
    }
  ];

  const futureFeatures = [
    {
      icon: <Palette className="w-6 h-6" />,
      title: "Emotion-Driven Narratives",
      description: "Stories that adapt to and evoke specific emotional responses"
    },
    {
      icon: <BookOpen className="w-6 h-6" />,
      title: "Multi-Modal Storytelling",
      description: "Integration of audio and visuals for immersive experiences"
    },
    {
      icon: <Sparkles className="w-6 h-6" />,
      title: "Expanding Universes",
      description: "Growing collection of characters, scenes, and interconnected stories"
    }
  ];

  return (
    <div className="min-h-screen bg-gray-900 py-24">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div 
          className="text-center mb-20"
          {...fadeIn}
        >
          <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
            Reimagining Storytelling
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Where ancient tradition meets artificial intelligence
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8 mb-20">
          {sections.map((section, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.2 }}
              className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 hover:transform hover:scale-105 transition-all"
            >
              <div className={`bg-gradient-to-r ${section.color} p-3 rounded-lg w-fit mb-6`}>
                {section.icon}
              </div>
              <h3 className="text-xl font-bold text-white mb-4">{section.title}</h3>
              <p className="text-gray-300">{section.content}</p>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 rounded-2xl p-12 backdrop-blur-sm mb-20"
        >
          <h2 className="text-3xl font-bold text-white mb-6">Our Vision</h2>
          <p className="text-lg text-gray-300 leading-relaxed">
            In the digital age, we're witnessing a transformation in how stories are told and experienced. 
            Our system combines the timeless art of storytelling with cutting-edge AI capabilities, 
            creating an adaptive platform that serves entertainment, education, and marketing needs. 
            Through live feedback and intuitive interfaces, we're building a system that doesn't just 
            tell stories - it creates worlds, builds relationships, and inspires imagination.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
        >
          <h2 className="text-3xl font-bold text-white mb-12 text-center">Future Horizons</h2>
          <div className="grid md:grid-cols-3 gap-8">
            {futureFeatures.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 1 + index * 0.2 }}
                className="bg-gray-800/30 backdrop-blur-sm rounded-xl p-6 hover:bg-gray-800/50 transition-all"
              >
                <div className="text-purple-400 mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default About; 