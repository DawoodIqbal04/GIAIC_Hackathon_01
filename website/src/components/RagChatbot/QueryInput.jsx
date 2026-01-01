import React, { useState } from 'react';
import './QueryInput.css';

const QueryInput = ({ onSubmit, disabled = false }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !disabled) {
      onSubmit(query.trim());
      setQuery('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form className="query-input-form" onSubmit={handleSubmit}>
      <div className="input-container">
        <textarea
          className="query-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about the book content..."
          disabled={disabled}
          rows="1"
          autoFocus
        />
        <button 
          type="submit" 
          className="submit-button" 
          disabled={disabled || !query.trim()}
        >
          {disabled ? (
            <span className="loading-spinner">●●●</span>
          ) : (
            'Send'
          )}
        </button>
      </div>
      <div className="input-hint">
        Press Enter to submit, Shift+Enter for a new line
      </div>
    </form>
  );
};

export default QueryInput;