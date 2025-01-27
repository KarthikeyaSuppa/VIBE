import { useState } from 'react';
import type { FC } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles } from 'lucide-react';
import StoryDialog from './StoryDialog';

const Hero: FC = () => {
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-b from-gray-900 to-purple-900">
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80')] bg-cover bg-center opacity-20" />
        <div className="absolute inset-0 bg-gradient-to-t from-gray-900 via-gray-900/40" />
      </div>
      
      <div className="relative max-w-7xl mx-auto px-6">
        <AnimatePresence>
          {!isDialogOpen && (
            <motion.div
              initial={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -100 }}
              className="py-32 text-center"
            >
              <motion.h1 
                className="text-5xl md:text-7xl font-bold text-white mb-6"
              >
                Create Magical Stories
              </motion.h1>
              
              <motion.p
                className="text-xl text-gray-300 mb-12 max-w-3xl mx-auto"
              >
                We humans tell stories from our past experiences. These stories contain multiple characters 
                and scenes, each evoking different emotions, and all centered around a unique theme. 
                Our mission is to recreate these experiences and generate new, immersive stories that 
                captivate and inspire.
              </motion.p>
              
              <motion.button
                onClick={() => setIsDialogOpen(true)}
                className="px-8 py-4 bg-purple-600 hover:bg-purple-700 text-white rounded-lg flex items-center justify-center space-x-2 transform hover:scale-105 transition-all mx-auto"
              >
                <Sparkles className="w-5 h-5" />
                <span>Start Creating</span>
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {isDialogOpen && (
            <motion.div
              initial={{ opacity: 0, y: 100 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 100 }}
              className="py-12"
            >
              <StoryDialog onClose={() => setIsDialogOpen(false)} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Hero;