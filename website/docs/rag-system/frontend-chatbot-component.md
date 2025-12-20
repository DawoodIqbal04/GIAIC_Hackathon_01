# Frontend Chatbot Interface Component

## Overview
This document outlines the development of the frontend chatbot interface component that will be integrated into the Docusaurus documentation site. The component will provide users with an interactive way to ask questions about the content and receive contextual responses.

## Component Design

### User Interface Elements

1. **Chat Window**: Collapsible panel that appears on all pages
2. **Message History**: Display of conversation with the AI assistant
3. **Input Area**: Text input with send button and optional voice input
4. **Source Citations**: Links to original documentation sources
5. **Feedback Controls**: Like/dislike buttons for responses

### Visual Design Principles

- Clean, modern interface that matches Docusaurus styling
- Non-intrusive design that doesn't interfere with documentation reading
- Responsive design that works on all device sizes
- Accessibility compliance (WCAG 2.1 AA)

## Component Implementation

### React Component Structure

Create `src/components/Chatbot.jsx`:

```jsx
import React, { useState, useEffect, useRef } from 'react';
import './Chatbot.css';

const Chatbot = ({ initialContext = {} }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [error, setError] = useState(null);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Initialize conversation
  useEffect(() => {
    if (isOpen && !conversationId) {
      createConversation();
    }
  }, [isOpen]);

  const createConversation = async () => {
    try {
      const response = await fetch('/api/conversation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          initialMessage: 'Hello, I need help with the documentation.',
          context: initialContext
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setConversationId(data.conversationId);
    } catch (err) {
      setError('Failed to initialize chat. Please try again later.');
      console.error('Conversation initialization error:', err);
    }
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputValue,
      timestamp: new Date().toISOString(),
    };

    // Add user message to UI immediately
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputValue,
          conversationId,
          context: initialContext,
          history: messages.slice(-6) // Send last 6 messages as context
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response,
        sources: data.sources || [],
        timestamp: data.timestamp,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setError('Failed to get response. Please try again.');
      console.error('Send message error:', err);
      
      // Add error message to chat
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString(),
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
    if (!isOpen && !conversationId) {
      createConversation();
    }
  };

  const clearChat = () => {
    setMessages([]);
    createConversation();
  };

  return (
    <div className="chatbot-container">
      {/* Chat toggle button */}
      <button 
        className={`chatbot-toggle ${isOpen ? 'open' : ''}`}
        onClick={toggleChat}
        aria-label={isOpen ? "Close chat" : "Open chat"}
      >
        {isOpen ? (
          <span className="close-icon">✕</span>
        ) : (
          <span className="chat-icon">💬</span>
        )}
      </button>

      {/* Chat window */}
      {isOpen && (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <div className="chatbot-title">Documentation Assistant</div>
            <div className="chatbot-controls">
              <button 
                className="clear-btn" 
                onClick={clearChat}
                title="Start new conversation"
              >
                🔄
              </button>
              <button 
                className="close-btn" 
                onClick={toggleChat}
                title="Close chat"
              >
                ✕
              </button>
            </div>
          </div>

          <div className="chatbot-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <p>Hello! I'm your documentation assistant.</p>
                <p>Ask me anything about the Physical AI and Humanoid Robotics Technical Book.</p>
              </div>
            ) : (
              messages.map((message) => (
                <div 
                  key={message.id} 
                  className={`message ${message.role}`}
                >
                  <div className="message-content">
                    {message.content}
                  </div>
                  
                  {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
                    <div className="sources">
                      <div className="sources-label">Sources:</div>
                      <ul className="sources-list">
                        {message.sources.slice(0, 3).map((source, idx) => (
                          <li key={idx} className="source-item">
                            <a 
                              href={`/docs/${source.document}`} 
                              target="_blank" 
                              rel="noopener noreferrer"
                            >
                              {source.module} - {source.chapter}
                            </a>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {message.role === 'assistant' && (
                    <div className="feedback-controls">
                      <button className="feedback-btn like" title="Helpful">👍</button>
                      <button className="feedback-btn dislike" title="Not helpful">👎</button>
                    </div>
                  )}
                </div>
              ))
            )}
            {isLoading && (
              <div className="message assistant">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chatbot-input-area">
            {error && <div className="error-message">{error}</div>}
            <div className="input-container">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about the documentation..."
                rows="1"
                disabled={isLoading}
                aria-label="Type your question"
              />
              <button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading}
                className="send-button"
                aria-label="Send message"
              >
                {isLoading ? '⏳' : '➤'}
              </button>
            </div>
            <div className="input-hint">
              Tip: Ask specific questions about ROS 2, Isaac, navigation, or humanoid robotics
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
```

### CSS Styling

Create `src/components/Chatbot.css`:

```css
.chatbot-container {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 1000;
  font-family: var(--ifm-font-family-base);
}

.chatbot-toggle {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  border: none;
  background: var(--ifm-color-primary);
  color: white;
  font-size: 24px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transition: all 0.3s ease;
}

.chatbot-toggle:hover {
  transform: scale(1.05);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
}

.chatbot-toggle.open {
  border-radius: 12px;
  width: 400px;
  height: 60px;
  justify-content: space-between;
  padding: 0 15px;
}

.chatbot-toggle.open .chat-icon {
  display: none;
}

.chatbot-toggle.open::before {
  content: 'Documentation Assistant';
  color: white;
  font-weight: 500;
}

.chatbot-window {
  width: 400px;
  height: 500px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.chatbot-header {
  background: var(--ifm-color-primary);
  color: white;
  padding: 15px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chatbot-title {
  font-weight: 600;
  font-size: 16px;
}

.chatbot-controls {
  display: flex;
  gap: 10px;
}

.chatbot-controls button {
  background: none;
  border: none;
  color: white;
  cursor: pointer;
  font-size: 16px;
  padding: 4px;
  border-radius: 4px;
}

.chatbot-controls button:hover {
  background: rgba(255, 255, 255, 0.2);
}

.chatbot-messages {
  flex: 1;
  overflow-y: auto;
  padding: 15px;
  display: flex;
  flex-direction: column;
  gap: 15px;
  background: #f9f9f9;
}

.welcome-message {
  text-align: center;
  color: #666;
  font-style: italic;
  padding: 20px 0;
}

.message {
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 18px;
  position: relative;
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.message.user {
  align-self: flex-end;
  background: var(--ifm-color-primary);
  color: white;
  border-bottom-right-radius: 4px;
}

.message.assistant {
  align-self: flex-start;
  background: white;
  color: #333;
  border-bottom-left-radius: 4px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.sources {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #eee;
  font-size: 0.85em;
}

.sources-label {
  font-weight: 600;
  color: #666;
  margin-bottom: 4px;
}

.sources-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.source-item {
  margin: 4px 0;
}

.source-item a {
  color: var(--ifm-color-primary);
  text-decoration: none;
}

.source-item a:hover {
  text-decoration: underline;
}

.feedback-controls {
  margin-top: 8px;
  display: flex;
  gap: 8px;
}

.feedback-btn {
  background: none;
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 4px 8px;
  cursor: pointer;
  font-size: 0.9em;
}

.feedback-btn:hover {
  border-color: var(--ifm-color-primary);
  color: var(--ifm-color-primary);
}

.typing-indicator {
  display: flex;
  align-items: center;
  padding: 8px 0;
}

.typing-indicator span {
  height: 8px;
  width: 8px;
  background: #999;
  border-radius: 50%;
  display: inline-block;
  margin: 0 2px;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

@keyframes typing {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

.chatbot-input-area {
  padding: 15px;
  background: white;
  border-top: 1px solid #eee;
}

.error-message {
  color: #e53e3e;
  font-size: 0.85em;
  margin-bottom: 10px;
  padding: 5px;
  background: #fef2f2;
  border-radius: 4px;
}

.input-container {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
}

.input-container textarea {
  flex: 1;
  border: 1px solid #ddd;
  border-radius: 18px;
  padding: 12px 16px;
  resize: none;
  font-family: inherit;
  font-size: 1em;
  outline: none;
  max-height: 100px;
}

.input-container textarea:focus {
  border-color: var(--ifm-color-primary);
}

.send-button {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: none;
  background: var(--ifm-color-primary);
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
}

.send-button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.input-hint {
  font-size: 0.75em;
  color: #888;
  text-align: center;
}

/* Responsive design */
@media (max-width: 480px) {
  .chatbot-container {
    bottom: 10px;
    right: 10px;
  }
  
  .chatbot-toggle {
    width: 50px;
    height: 50px;
  }
  
  .chatbot-toggle.open {
    width: 300px;
    height: 50px;
  }
  
  .chatbot-window {
    width: 300px;
    height: 400px;
  }
  
  .message {
    max-width: 90%;
  }
}
```

## Integration with Docusaurus

### 1. Add Component to Layout

Update `src/theme/Layout/index.js` to include the chatbot on all pages:

```jsx
import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import Chatbot from '@site/src/components/Chatbot';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props}>
        {props.children}
        <Chatbot />
      </OriginalLayout>
    </>
  );
}
```

### 2. Create a Docusaurus Plugin

Create `plugins/docusaurus-chatbot-plugin/index.js`:

```javascript
const path = require('path');

module.exports = function(context, options) {
  return {
    name: 'docusaurus-chatbot-plugin',
    
    getThemePath() {
      return path.resolve(__dirname, './theme');
    },
    
    getClientModules() {
      return [path.resolve(__dirname, './client/chatbot-client.js')];
    },
    
    configureWebpack(config, isServer, utils) {
      return {
        resolve: {
          alias: {
            '@chatbot': path.resolve(__dirname, '../src/components/Chatbot'),
          },
        },
      };
    },
  };
};
```

### 3. Add Plugin to Docusaurus Config

Update `docusaurus.config.js`:

```javascript
module.exports = {
  // ... other config
  plugins: [
    // ... other plugins
    [
      path.resolve(__dirname, 'plugins/docusaurus-chatbot-plugin'),
      {
        // plugin options
      }
    ],
  ],
  // ... rest of config
};
```

## Advanced Features

### Context-Aware Responses

Enhance the component to provide context based on the current page:

```jsx
import { useLocation } from '@docusaurus/router';

const Chatbot = () => {
  const location = useLocation();
  
  // Extract context from current page
  useEffect(() => {
    const pathParts = location.pathname.split('/');
    const module = pathParts[2]; // Assuming URL structure is /docs/module/chapter
    const chapter = pathParts[3];
    
    const context = {
      currentModule: module,
      currentChapter: chapter,
      currentPage: location.pathname
    };
    
    // Pass context to chatbot
    setInitialContext(context);
  }, [location.pathname]);
  
  // ... rest of component
};
```

### Voice Input Capability

Add voice input functionality:

```jsx
import { useSpeechRecognition } from 'react-speech-recognition';

const Chatbot = () => {
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition();

  if (!browserSupportsSpeechRecognition) {
    return <span>Browser doesn't support speech recognition.</span>;
  }

  const startListening = () => {
    resetTranscript();
    SpeechRecognition.startListening();
  };

  return (
    <div className="input-container">
      <textarea
        value={inputValue || transcript}
        onChange={(e) => setInputValue(e.target.value)}
        // ... other props
      />
      <button 
        onClick={startListening}
        className={`voice-btn ${listening ? 'active' : ''}`}
      >
        {listening ? '🔴' : '🎤'}
      </button>
      <button onClick={sendMessage} className="send-button">
        ➤
      </button>
    </div>
  );
};
```

## Accessibility Features

1. **Screen Reader Support**: Proper ARIA labels and roles
2. **Keyboard Navigation**: Full functionality via keyboard
3. **Color Contrast**: Meets WCAG 2.1 AA standards
4. **Focus Management**: Clear focus indicators and logical focus order

## Performance Optimization

1. **Code Splitting**: Lazy load chatbot component
2. **Message Limiting**: Limit message history to prevent memory issues
3. **Debouncing**: Prevent rapid API requests
4. **Caching**: Cache API responses where appropriate

This implementation provides a fully functional, accessible, and well-designed chatbot interface that integrates seamlessly with the Docusaurus documentation site.