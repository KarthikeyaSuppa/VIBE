import React from 'react';

const FloatingElements = () => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <div className="absolute w-64 h-64 bg-purple-500/10 rounded-full blur-3xl top-20 -left-32 animate-float" />
      <div className="absolute w-96 h-96 bg-blue-500/10 rounded-full blur-3xl -top-48 right-0 animate-float-delayed" />
      <div className="absolute w-48 h-48 bg-pink-500/10 rounded-full blur-3xl bottom-20 right-32 animate-float" />
    </div>
  );
};

export default FloatingElements;