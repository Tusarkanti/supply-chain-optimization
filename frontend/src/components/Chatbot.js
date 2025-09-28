import React, { useState } from 'react';
import './Chatbot.css';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleChatbot = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {/* Floating Chatbot Button */}
      <div className="chatbot-button" onClick={toggleChatbot}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2C6.48 2 2 6.48 2 12c0 1.54.36 3.05 1.05 4.42L2 22l5.58-1.05C9.95 21.64 11.46 22 13 22h7c1.1 0 2-.9 2-2V12c0-5.52-4.48-10-10-10z" fill="white"/>
          <circle cx="8" cy="12" r="1" fill="#1976d2"/>
          <circle cx="12" cy="12" r="1" fill="#1976d2"/>
          <circle cx="16" cy="12" r="1" fill="#1976d2"/>
        </svg>
      </div>

      {/* Chatbot Modal */}
      {isOpen && (
        <div className="chatbot-modal">
          <div className="chatbot-header">
            <h3>Supply Chain Assistant</h3>
            <button className="chatbot-close" onClick={toggleChatbot}>×</button>
          </div>
          <div className="chatbot-content">
            <iframe
              src="https://creator.voiceflow.com/share/68d624e146d02ffa8bb88355/development"
              width="100%"
              height="100%"
              frameBorder="0"
              allow="microphone"
              title="Voiceflow Chatbot"
            />
          </div>
        </div>
      )}

      {/* Modal Overlay */}
      {isOpen && <div className="chatbot-overlay" onClick={toggleChatbot}></div>}
    </>
  );
};

export default Chatbot;
