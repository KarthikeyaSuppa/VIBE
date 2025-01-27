import React from 'react';
import { Sparkles, Menu } from 'lucide-react';

const Navbar = () => {
  return (
    <nav className="fixed w-full bg-transparent backdrop-blur-sm z-50 px-6 py-4">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-6 h-6 text-purple-500" />
          <span className="text-2xl font-bold bg-gradient-to-r from-purple-500 to-blue-500 bg-clip-text text-transparent">
            VIBE
          </span>
        </div>
        <div className="hidden md:flex space-x-8">
          <a href="/" className="text-gray-200 hover:text-purple-400 transition-colors">Home</a>
          <a href="/help" className="text-gray-200 hover:text-purple-400 transition-colors">Help</a>
          <a href="/about" className="text-gray-200 hover:text-purple-400 transition-colors">About</a>
          <a href="/contact" className="text-gray-200 hover:text-purple-400 transition-colors">Contact</a>
        </div>
        <button className="md:hidden text-gray-200">
          <Menu className="w-6 h-6" />
        </button>
      </div>
    </nav>
  );
};

export default Navbar;