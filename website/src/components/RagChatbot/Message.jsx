import React from 'react';
import './Message.css';

const Message = ({ message }) => {
  const { type, content, citations, timestamp } = message;

  // Format timestamp for display
  const formatTime = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Render citations if they exist
  const renderCitations = () => {
    if (!citations || citations.length === 0) return null;

    return (
      <div className="citations">
        <h4>Sources:</h4>
        <ul>
          {citations.map((citation, index) => (
            <li key={index} className="citation-item">
              <strong>{citation.source_reference}</strong>
              <p className="citation-snippet">"{citation.content_snippet}"</p>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  // Determine message class based on type
  const messageClass = `message ${type}-message`;

  return (
    <div className={messageClass}>
      <div className="message-content">
        <p>{content}</p>
        {type === 'bot' && renderCitations()}
      </div>
      {timestamp && (
        <div className="message-timestamp">
          {formatTime(timestamp)}
        </div>
      )}
    </div>
  );
};

export default Message;