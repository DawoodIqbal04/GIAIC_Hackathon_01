# Integration of Chatbot with Docusaurus Documentation Site

## Overview
This document outlines the complete integration process of the RAG chatbot system with the Docusaurus documentation site. The integration includes frontend components, backend API connections, and content indexing mechanisms.

## Integration Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Docusaurus     │    │  Chatbot API     │    │  Vector DB &    │
│  Documentation  │───▶│  (Backend)       │───▶│  LLM Service    │
│  Site           │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ Frontend Chat   │
                       │ Component       │
                       └─────────────────┘
```

## Step-by-Step Integration Process

### 1. Backend API Setup

#### A. Install Dependencies
```bash
cd website
npm install express cors weaviate-client openai dotenv axios
npm install -D nodemon concurrently
```

#### B. Create API Server
Create `api-server.js`:

```javascript
const express = require('express');
const cors = require('cors');
const weaviate = require('weaviate-client');
const { Configuration, OpenAIApi } = require('openai');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Initialize Weaviate client
const weaviateClient = weaviate.client({
  scheme: process.env.WEAVIATE_SCHEME || 'http',
  host: process.env.WEAVIATE_HOST || 'localhost:8080',
});

// Initialize OpenAI client
const openai = new OpenAIApi(
  new Configuration({
    apiKey: process.env.OPENAI_API_KEY,
  })
);

// In-memory storage for conversations (use Redis or DB in production)
const conversations = new Map();

// Import API routes from separate files
const searchRoutes = require('./routes/search');
const chatRoutes = require('./routes/chat');
const conversationRoutes = require('./routes/conversation');
const feedbackRoutes = require('./routes/feedback');

app.use('/api/search', searchRoutes);
app.use('/api/chat', chatRoutes);
app.use('/api/conversation', conversationRoutes);
app.use('/api/feedback', feedbackRoutes);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

app.listen(PORT, () => {
  console.log(`Chatbot API server running on port ${PORT}`);
});
```

#### C. Create API Route Files

Create `routes/search.js`:

```javascript
const express = require('express');
const router = express.Router();
const weaviate = require('weaviate-client');

const weaviateClient = weaviate.client({
  scheme: process.env.WEAVIATE_SCHEME || 'http',
  host: process.env.WEAVIATE_HOST || 'localhost:8080',
});

router.post('/', async (req, res) => {
  try {
    const { query, limit = 5, filters = {} } = req.body;
    
    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }
    
    // Build the GraphQL query for Weaviate
    let graphQLQuery = `
      {
        Get {
          DocChunk(
            nearText: {
              concepts: ["${query}"]
            }
            limit: ${limit}
    `;
    
    // Add filters if provided
    if (filters.module) {
      graphQLQuery += `
            where: {
              path: ["module"]
              operator: Equal
              valueString: "${filters.module}"
            }
      `;
    }
    
    graphQLQuery += `
          ) {
            content
            sourceDocument
            sectionTitle
            module
            chapter
            _additional {
              id
              certainty
            }
          }
        }
      }
    `;
    
    // Execute the query
    const result = await weaviateClient.graphql.raw(graphQLQuery);
    
    const results = result.data.Get.DocChunk.map(item => ({
      id: item._additional.id,
      content: item.content,
      sourceDocument: item.sourceDocument,
      sectionTitle: item.sectionTitle,
      module: item.module,
      chapter: item.chapter,
      similarity: item._additional.certainty
    }));
    
    res.json({
      query,
      results,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).json({ error: 'Search failed' });
  }
});

module.exports = router;
```

Create `routes/chat.js`:

```javascript
const express = require('express');
const router = express.Router();
const weaviate = require('weaviate-client');
const { Configuration, OpenAIApi } = require('openai');

const weaviateClient = weaviate.client({
  scheme: process.env.WEAVIATE_SCHEME || 'http',
  host: process.env.WEAVIATE_HOST || 'localhost:8080',
});

const openai = new OpenAIApi(
  new Configuration({
    apiKey: process.env.OPENAI_API_KEY,
  })
);

// In-memory storage for conversations (use Redis or DB in production)
const conversations = new Map();

router.post('/', async (req, res) => {
  try {
    const { 
      message, 
      conversationId, 
      context = {}, 
      history = [] 
    } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    // Perform semantic search to find relevant context
    const searchResponse = await fetch(`${process.env.API_BASE_URL || 'http://localhost:3001'}/api/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        query: message, 
        limit: 5,
        filters: context 
      })
    });
    
    const searchResult = await searchResponse.json();
    const relevantChunks = searchResult.results;
    
    // Format the context for the LLM
    const contextText = relevantChunks
      .map(chunk => `Source: ${chunk.sourceDocument} | ${chunk.content}`)
      .join('\n\n');
    
    // Prepare the prompt for the LLM
    const chatHistory = history.map(msg => `${msg.role}: ${msg.content}`).join('\n');
    const prompt = `
      You are an expert assistant for the Physical AI and Humanoid Robotics Technical Book.
      Use the following context to answer the user's question.
      If the context doesn't contain the information, say so.
      
      Context:
      ${contextText}
      
      Previous conversation:
      ${chatHistory}
      
      User: ${message}
      Assistant:`;
    
    // Call the LLM to generate a response
    const completion = await openai.createChatCompletion({
      model: 'gpt-3.5-turbo',
      messages: [
        { 
          role: 'system', 
          content: 'You are an expert assistant for the Physical AI and Humanoid Robotics Technical Book. Provide accurate, helpful responses based on the provided context.' 
        },
        { 
          role: 'user', 
          content: prompt 
        }
      ],
      max_tokens: 500,
      temperature: 0.7
    });
    
    const responseText = completion.data.choices[0].message.content;
    
    // Create or update conversation
    if (!conversations.has(conversationId)) {
      conversations.set(conversationId, []);
    }
    
    // Add the exchange to the conversation
    const conversation = conversations.get(conversationId);
    conversation.push(
      { role: 'user', content: message, timestamp: new Date().toISOString() },
      { 
        role: 'assistant', 
        content: responseText, 
        sources: relevantChunks,
        timestamp: new Date().toISOString() 
      }
    );
    
    // Return the response with sources
    res.json({
      conversationId,
      response: responseText,
      sources: relevantChunks.map(chunk => ({
        document: chunk.sourceDocument,
        section: chunk.sectionTitle,
        module: chunk.module,
        chapter: chunk.chapter,
        similarity: chunk.similarity
      })),
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ error: 'Chat request failed' });
  }
});

module.exports = router;
```

### 2. Frontend Component Integration

#### A. Create the Chatbot Component

Create `src/components/Chatbot/Chatbot.jsx`:

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

#### B. Add CSS Styling

Create `src/components/Chatbot/Chatbot.css`:

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

### 3. Integrate Component into Docusaurus Layout

#### A. Create Layout Wrapper

Create `src/theme/Layout/index.js`:

```jsx
import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import Chatbot from '@site/src/components/Chatbot/Chatbot';

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

### 4. Update Package.json Scripts

Add to `package.json`:

```json
{
  "scripts": {
    "docusaurus": "docusaurus",
    "start": "concurrently \"npm run api-server\" \"docusaurus start\"",
    "build": "docusaurus build",
    "swizzle": "docusaurus swizzle",
    "deploy": "docusaurus deploy",
    "clear": "docusaurus clear",
    "api-server": "node api-server.js",
    "dev": "concurrently \"npm run api-server\" \"docusaurus start\""
  }
}
```

### 5. Environment Configuration

Create `.env` file in the website directory:

```
# Weaviate Configuration
WEAVIATE_SCHEME=http
WEAVIATE_HOST=localhost:8080

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Server Configuration
PORT=3001
API_BASE_URL=http://localhost:3001

# Rate Limiting
RATE_LIMIT_WINDOW_MS=900000  # 15 minutes
RATE_LIMIT_MAX_REQUESTS=100
```

### 6. Content Indexing Integration

#### A. Update Build Process

Modify `docusaurus.config.js` to include content indexing:

```javascript
// docusaurus.config.js
const config = {
  // ... other config options

  scripts: [
    // ... other scripts
  ],
  
  plugins: [
    // ... other plugins
    
    // Add a plugin to run content indexing during build
    async function contentIndexingPlugin(context, options) {
      return {
        name: 'content-indexing-plugin',
        async postBuild(props) {
          console.log('Starting content indexing after build...');
          
          // Import and run the indexing function
          const { indexAllDocs } = require('./scripts/index-content');
          await indexAllDocs();
          
          console.log('Content indexing completed!');
        },
      };
    },
  ],
  
  // ... rest of config
};

module.exports = config;
```

#### B. Create Content Indexing Script

Create `scripts/index-content.js`:

```javascript
const fs = require('fs');
const path = require('path');
const glob = require('glob');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const weaviate = require('weaviate-client');

// Configuration
const DOCS_DIR = './docs';
const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 100;

// Initialize Weaviate client
const client = weaviate.client({
  scheme: process.env.WEAVIATE_SCHEME || 'http',
  host: process.env.WEAVIATE_HOST || 'localhost:8080',
});

// Text splitter for chunking content
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
});

/**
 * Extract metadata from markdown file
 */
function extractMetadata(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  
  // Extract frontmatter if present
  const frontmatterRegex = /^---\s*\n(.*?)\n---\s*\n/s;
  const frontmatterMatch = content.match(frontmatterRegex);
  
  let metadata = {};
  if (frontmatterMatch) {
    const frontmatter = frontmatterMatch[1];
    // Simple frontmatter parser (in production, use a proper YAML parser)
    const lines = frontmatter.split('\n');
    lines.forEach(line => {
      const [key, value] = line.split(': ');
      if (key && value) {
        metadata[key.trim()] = value.trim().replace(/^["']|["']$/g, '');
      }
    });
  }
  
  // Extract module and chapter from path
  const pathParts = filePath.split(path.sep);
  const moduleIndex = pathParts.indexOf('docs') + 1;
  if (moduleIndex < pathParts.length) {
    metadata.module = pathParts[moduleIndex];
    
    if (moduleIndex + 1 < pathParts.length) {
      const chapterPart = pathParts[moduleIndex + 1];
      if (chapterPart.includes('chapter')) {
        metadata.chapter = chapterPart;
      }
    }
  }
  
  return metadata;
}

/**
 * Extract content from markdown file, excluding frontmatter
 */
function extractContent(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  
  // Remove frontmatter if present
  const frontmatterRegex = /^---\s*\n(.*?)\n---\s*\n/s;
  const contentWithoutFrontmatter = content.replace(frontmatterRegex, '');
  
  // Remove markdown syntax for better semantic search
  // This is a simplified version - in production, use a proper markdown parser
  return contentWithoutFrontmatter
    .replace(/#{1,6}\s+/g, '') // Remove headers
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove links, keep text
    .replace(/\*\*([^*]+)\*\*/g, '$1') // Remove bold
    .replace(/\*([^*]+)\*/g, '$1') // Remove italic
    .replace(/!\[[^\]]*\]\([^)]+\)/g, '') // Remove image tags
    .replace(/`([^`]+)`/g, '$1') // Remove inline code
    .replace(/```[\s\S]*?```/g, '') // Remove code blocks
    .replace(/\n{3,}/g, '\n\n') // Normalize multiple newlines
    .trim();
}

/**
 * Process a single markdown file
 */
async function processFile(filePath) {
  console.log(`Processing file: ${filePath}`);
  
  try {
    const metadata = extractMetadata(filePath);
    const content = extractContent(filePath);
    
    // Split content into chunks
    const chunks = await textSplitter.splitText(content);
    
    // Get the section title from the first H1 or H2 in the content
    const titleMatch = content.match(/^(#{1,2}\s+.*$)/m);
    const sectionTitle = titleMatch ? titleMatch[1].replace(/#{1,2}\s+/, '').trim() : 'Untitled';
    
    // Index each chunk
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      
      // Skip empty chunks
      if (!chunk.trim()) continue;
      
      // Create data object for Weaviate
      const dataObj = {
        content: chunk,
        sourceDocument: path.basename(filePath),
        sectionTitle: sectionTitle,
        module: metadata.module || 'Unknown Module',
        chapter: metadata.chapter || 'Unknown Chapter',
        chunkIndex: i
      };
      
      // Add to Weaviate
      await client.data
        .creator()
        .withClassName('DocChunk')
        .withProperties(dataObj)
        .do();
        
      console.log(`  Indexed chunk ${i+1}/${chunks.length} from ${path.basename(filePath)}`);
    }
    
    console.log(`Finished processing: ${filePath}`);
  } catch (error) {
    console.error(`Error processing file ${filePath}:`, error);
  }
}

/**
 * Index all markdown files in the docs directory
 */
async function indexAllDocs() {
  console.log('Starting content indexing...');
  
  // Find all markdown files in docs directory
  const pattern = path.join(DOCS_DIR, '**/*.md');
  const files = glob.sync(pattern, {
    ignore: ['**/node_modules/**', '**/build/**']
  });
  
  console.log(`Found ${files.length} markdown files to process`);
  
  // Process each file
  for (const file of files) {
    await processFile(file);
  }
  
  console.log('Content indexing completed!');
}

/**
 * Clear existing content from Weaviate before re-indexing
 */
async function clearExistingContent() {
  console.log('Clearing existing content from Weaviate...');
  
  try {
    await client.batch
      .objectsBatchDeleter()
      .withClassName('DocChunk')
      .do();
      
    console.log('Existing content cleared');
  } catch (error) {
    console.error('Error clearing existing content:', error);
  }
}

// Main execution
async function main() {
  // Clear existing content first
  await clearExistingContent();
  
  // Index all documentation
  await indexAllDocs();
}

// Run if this file is executed directly
if (require.main === module) {
  main().catch(console.error);
}

module.exports = {
  extractMetadata,
  extractContent,
  processFile,
  indexAllDocs
};
```

### 7. Testing the Integration

Create a test script `test-integration.js`:

```javascript
const axios = require('axios');

async function testIntegration() {
  console.log('Testing RAG chatbot integration...');
  
  try {
    // Test API server health
    console.log('1. Testing API server health...');
    const healthRes = await axios.get('http://localhost:3001/health');
    console.log('   ✓ API server is healthy:', healthRes.data.status);
    
    // Test content search
    console.log('2. Testing content search...');
    const searchRes = await axios.post('http://localhost:3001/api/search', {
      query: 'ROS 2 middleware',
      limit: 3
    });
    console.log('   ✓ Search returned', searchRes.data.results.length, 'results');
    
    // Test conversation creation
    console.log('3. Testing conversation creation...');
    const convRes = await axios.post('http://localhost:3001/api/conversation', {
      initialMessage: 'Hello, I need help with ROS 2'
    });
    const convId = convRes.data.conversationId;
    console.log('   ✓ Conversation created:', convId);
    
    // Test chat functionality
    console.log('4. Testing chat functionality...');
    const chatRes = await axios.post('http://localhost:3001/api/chat', {
      message: 'What is ROS 2 middleware?',
      conversationId: convId
    });
    console.log('   ✓ Received response with', chatRes.data.sources.length, 'sources');
    
    console.log('\n✓ All integration tests passed!');
  } catch (error) {
    console.error('✗ Integration test failed:', error.message);
  }
}

if (require.main === module) {
  testIntegration();
}
```

## Deployment Considerations

### 1. Production Setup
- Use a production-grade vector database (Pinecone, Weaviate cloud)
- Implement proper authentication and rate limiting
- Set up monitoring and logging
- Configure SSL for secure communication

### 2. Performance Optimization
- Implement caching for frequent queries
- Use CDN for static assets
- Optimize component loading with code splitting
- Implement message history limits

### 3. Scalability
- Use Redis for conversation storage
- Implement load balancing for API servers
- Consider microservice architecture for larger deployments
- Set up auto-scaling based on traffic

This comprehensive integration ensures that the RAG chatbot is fully embedded in the Docusaurus documentation site, providing users with an interactive way to explore and understand the content.