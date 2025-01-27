import { motion } from 'framer-motion';
import { Github, Linkedin, Mail, Star } from 'lucide-react';

const Contact = () => {
  const fadeIn = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 }
  };

  const teamLead = {
    name: "Karthikeya Suppa",
    role: "Project Lead",
    description: "Research, Planning, Design and Execution Lead",
    links: {
      linkedin: "https://www.linkedin.com/in/karthikeyasuppa/",
      github: "https://github.com/KarthikeyaSuppa",
      email: "karthikeyasuppa@gmail.com"
    }
  };

  const coreMember = {
    name: "Byathala Teja",
    role: "Technical Lead & Design",
    email: "natateja1662@gmail.com"
  };

  const researchTeam = [
    {
      name: "Gudiboina Karthik",
      role: "Database Design & Research",
      email: "gkarthik.1673360@gmail.com"
    },
    {
      name: "Bharath Nandan Vadla",
      role: "Database Design & Research",
      email: "bharathkanna433@gmail.com"
    }
  ];

  const contributors = [
    {
      name: "C Dhanush",
      role: "Content Contributor",
      email: "danushdannu7@gmail.com"
    },
    {
      name: "Kavali Saivarun",
      role: "Content Contributor",
      email: "kavalisaivarun@gmail.com"
    },
    {
      name: "Vorsu Manoj",
      role: "Content Contributor",
      email: "vmanojkumar21165@gmail.com"
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
            Meet Our Team
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            The brilliant minds behind VIBE Storytelling
          </p>
        </motion.div>

        {/* Project Lead */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 rounded-2xl p-8 backdrop-blur-sm mb-12"
        >
          <div className="flex flex-col md:flex-row items-center md:items-start gap-8">
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-white mb-2">{teamLead.name}</h2>
              <p className="text-purple-400 mb-4">{teamLead.role}</p>
              <p className="text-gray-300 mb-6">{teamLead.description}</p>
              <div className="flex gap-4">
                <a
                  href={teamLead.links.linkedin}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-300 hover:text-purple-400 transition-colors"
                >
                  <Linkedin className="w-6 h-6" />
                </a>
                <a
                  href={teamLead.links.github}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-300 hover:text-purple-400 transition-colors"
                >
                  <Github className="w-6 h-6" />
                </a>
                <a
                  href={`mailto:${teamLead.links.email}`}
                  className="text-gray-300 hover:text-purple-400 transition-colors"
                >
                  <Mail className="w-6 h-6" />
                </a>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Core Team */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gray-800/50 rounded-xl p-8 mb-12"
        >
          <h2 className="text-2xl font-bold text-white mb-8">Core Team</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="p-6 bg-gray-700/30 rounded-lg">
              <h3 className="text-xl font-semibold text-white mb-2">{coreMember.name}</h3>
              <p className="text-purple-400 mb-4">{coreMember.role}</p>
              <a href={`mailto:${coreMember.email}`} className="text-gray-300 hover:text-purple-400 transition-colors flex items-center gap-2">
                <Mail className="w-5 h-5" />
                {coreMember.email}
              </a>
            </div>
            {researchTeam.map((member, index) => (
              <div key={index} className="p-6 bg-gray-700/30 rounded-lg">
                <h3 className="text-xl font-semibold text-white mb-2">{member.name}</h3>
                <p className="text-purple-400 mb-4">{member.role}</p>
                <a href={`mailto:${member.email}`} className="text-gray-300 hover:text-purple-400 transition-colors flex items-center gap-2">
                  <Mail className="w-5 h-5" />
                  {member.email}
                </a>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Contributors */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-gray-800/50 rounded-xl p-8 mb-12"
        >
          <h2 className="text-2xl font-bold text-white mb-8">Contributors</h2>
          <div className="grid md:grid-cols-3 gap-6">
            {contributors.map((contributor, index) => (
              <div key={index} className="p-6 bg-gray-700/30 rounded-lg">
                <h3 className="text-lg font-semibold text-white mb-2">{contributor.name}</h3>
                <p className="text-purple-400 mb-4">{contributor.role}</p>
                <a href={`mailto:${contributor.email}`} className="text-gray-300 hover:text-purple-400 transition-colors flex items-center gap-2">
                  <Mail className="w-5 h-5" />
                  {contributor.email}
                </a>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Project Guide */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="text-center bg-gray-800/30 rounded-xl p-8"
        >
          <Star className="w-8 h-8 text-yellow-400 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Special Thanks</h2>
          <p className="text-gray-300">
            To our project guide <span className="text-purple-400">Haindavi Ponguvala</span> for giving us 
            the freedom to explore and develop this project.
          </p>
        </motion.div>
      </div>
    </div>
  );
};

export default Contact; 