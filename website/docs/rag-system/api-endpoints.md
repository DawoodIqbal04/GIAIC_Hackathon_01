# API Endpoints for Chatbot Communication

## Overview
This document outlines the API endpoints required for the RAG chatbot system to communicate with the backend services. These endpoints will handle search queries, conversation management, and response generation.

## API Design Principles

- RESTful design with appropriate HTTP methods
- JSON request/response format
- Proper error handling and status codes
- Rate limiting and authentication considerations
- Consistent response structure

## API Endpoints

### 1. Search Endpoint
**URL**: `POST /api/search`
**Description**: Perform semantic search on the documentation content

**Request**:
```json
{
  "query": "What is ROS 2 middleware?",
  "limit": 5,
  "filters": {
    "module": "Module 1",
    "chapter": "Chapter 1"
  }
}
```

**Response**:
```json
{
  "query": "What is ROS 2 middleware?",
  "results": [
    {
      "id": "chunk_123",
      "content": "ROS 2 middleware provides the communication layer between different robot components...",
      "sourceDocument": "chapter-1-middleware-control.md",
      "sectionTitle": "Introduction to ROS 2 Middleware",
      "module": "Module 1: ROS 2 and Robotic Control Foundations",
      "chapter": "Chapter 1: Middleware for robot control",
      "similarity": 0.92
    }
  ],
  "timestamp": "2023-10-01T10:00:00Z"
}
```

### 2. Chat Endpoint
**URL**: `POST /api/chat`
**Description**: Process a chat message and return a contextual response

**Request**:
```json
{
  "message": "How do ROS 2 nodes communicate?",
  "conversationId": "conv_123",
  "context": {
    "module": "Module 1",
    "chapter": "Chapter 2"
  },
  "history": [
    {
      "role": "user",
      "content": "What is ROS 2?"
    },
    {
      "role": "assistant",
      "content": "ROS 2 is the latest version of the Robot Operating System..."
    }
  ]
}
```

**Response**:
```json
{
  "conversationId": "conv_123",
  "response": "ROS 2 nodes communicate through topics, services, and actions...",
  "sources": [
    {
      "document": "chapter-2-nodes-topics-services.md",
      "section": "Communication Patterns",
      "module": "Module 1: ROS 2 and Robotic Control Foundations",
      "chapter": "Chapter 2: ROS 2 nodes, topics, and services",
      "similarity": 0.89
    }
  ],
  "timestamp": "2023-10-01T10:00:05Z"
}
```

### 3. Conversation Management Endpoints

#### Create Conversation
**URL**: `POST /api/conversation`
**Description**: Start a new conversation

**Request**:
```json
{
  "initialMessage": "I want to learn about ROS 2",
  "context": {
    "module": "Module 1",
    "preferredLanguage": "en"
  }
}
```

**Response**:
```json
{
  "conversationId": "conv_456",
  "timestamp": "2023-10-01T10:00:00Z",
  "context": {
    "module": "Module 1",
    "preferredLanguage": "en"
  }
}
```

#### Get Conversation History
**URL**: `GET /api/conversation/{conversationId}`
**Description**: Retrieve conversation history

**Response**:
```json
{
  "conversationId": "conv_456",
  "history": [
    {
      "id": "msg_1",
      "role": "user",
      "content": "I want to learn about ROS 2",
      "timestamp": "2023-10-01T10:00:00Z"
    },
    {
      "id": "msg_2", 
      "role": "assistant",
      "content": "ROS 2 is the latest version of the Robot Operating System...",
      "sources": [
        {
          "document": "module-1-intro.md",
          "section": "Introduction",
          "module": "Module 1: ROS 2 and Robotic Control Foundations"
        }
      ],
      "timestamp": "2023-10-01T10:00:02Z"
    }
  ]
}
```

### 4. Feedback Endpoint
**URL**: `POST /api/feedback`
**Description**: Submit feedback on a response

**Request**:
```json
{
  "conversationId": "conv_123",
  "messageId": "msg_456",
  "rating": 5,
  "comment": "Very helpful explanation",
  "isAccurate": true,
  "isRelevant": true
}
```

**Response**:
```json
{
  "status": "Feedback received",
  "timestamp": "2023-10-01T10:00:10Z"
}
```

## Implementation Example (Node.js/Express)

Create `api/chatbot.js`:

```javascript
const express = require('express');
const router = express.Router();
const weaviate = require('weaviate-client');
const { Configuration, OpenAIApi } = require('openai');

// Initialize Weaviate client
const weaviateClient = weaviate.client({
  scheme: process.env.WEAVIATE_SCHEME || 'http',
  host: process.env.WEAVIATE_HOST || 'localhost:8080',
});

// Initialize OpenAI client (or your preferred LLM service)
const openai = new OpenAIApi(
  new Configuration({
    apiKey: process.env.OPENAI_API_KEY,
  })
);

// In-memory storage for conversations (use Redis or DB in production)
const conversations = new Map();

// Search endpoint
router.post('/search', async (req, res) => {
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

// Chat endpoint
router.post('/chat', async (req, res) => {
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
    const searchResponse = await fetch(`${process.env.API_BASE_URL}/api/search`, {
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

// Create conversation endpoint
router.post('/conversation', (req, res) => {
  const { initialMessage, context = {} } = req.body;
  
  const conversationId = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  conversations.set(conversationId, []);
  
  if (initialMessage) {
    conversations.get(conversationId).push({
      role: 'user',
      content: initialMessage,
      timestamp: new Date().toISOString()
    });
  }
  
  res.json({
    conversationId,
    timestamp: new Date().toISOString(),
    context
  });
});

// Get conversation history endpoint
router.get('/conversation/:conversationId', (req, res) => {
  const { conversationId } = req.params;
  
  if (!conversations.has(conversationId)) {
    return res.status(404).json({ error: 'Conversation not found' });
  }
  
  res.json({
    conversationId,
    history: conversations.get(conversationId)
  });
});

// Feedback endpoint
router.post('/feedback', (req, res) => {
  const { conversationId, messageId, rating, comment, isAccurate, isRelevant } = req.body;
  
  // In a real implementation, store feedback in a database
  console.log('Feedback received:', {
    conversationId,
    messageId,
    rating,
    comment,
    isAccurate,
    isRelevant,
    timestamp: new Date().toISOString()
  });
  
  res.json({
    status: 'Feedback received',
    timestamp: new Date().toISOString()
  });
});

module.exports = router;
```

## Server Setup

Create the main server file `server.js`:

```javascript
const express = require('express');
const cors = require('cors');
const chatbotRoutes = require('./api/chatbot');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// API routes
app.use('/api', chatbotRoutes);

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

## Package Dependencies

Add to `package.json`:

```json
{
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "weaviate-client": "^2.1.0",
    "openai": "^3.3.0",
    "dotenv": "^16.3.1"
  },
  "scripts": {
    "api-server": "node server.js",
    "dev": "nodemon server.js"
  }
}
```

## Environment Configuration

Create `.env` file:

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

## Security Considerations

1. **Authentication**: Implement JWT or API key authentication for endpoints
2. **Rate Limiting**: Add rate limiting middleware to prevent abuse
3. **Input Validation**: Validate all inputs to prevent injection attacks
4. **CORS**: Configure CORS appropriately for your frontend domain
5. **HTTPS**: Use HTTPS in production

## Testing the API

Create a test file `test-api.js`:

```javascript
const axios = require('axios');

const API_BASE = 'http://localhost:3001/api';

async function testAPI() {
  try {
    // Test search endpoint
    console.log('Testing search endpoint...');
    const searchResponse = await axios.post(`${API_BASE}/search`, {
      query: 'ROS 2 middleware',
      limit: 3
    });
    console.log('Search results:', searchResponse.data.results.length, 'items');

    // Test conversation creation
    console.log('Testing conversation creation...');
    const convResponse = await axios.post(`${API_BASE}/conversation`, {
      initialMessage: 'What is ROS 2?'
    });
    const convId = convResponse.data.conversationId;
    console.log('Created conversation:', convId);

    // Test chat endpoint
    console.log('Testing chat endpoint...');
    const chatResponse = await axios.post(`${API_BASE}/chat`, {
      message: 'How do ROS 2 nodes communicate?',
      conversationId: convId
    });
    console.log('Chat response length:', chatResponse.data.response.length);

    // Test conversation history
    console.log('Testing conversation history...');
    const historyResponse = await axios.get(`${API_BASE}/conversation/${convId}`);
    console.log('History messages:', historyResponse.data.history.length);

  } catch (error) {
    console.error('API test error:', error.response?.data || error.message);
  }
}

if (require.main === module) {
  testAPI();
}
```

This implementation provides a complete set of API endpoints for the RAG chatbot system, allowing for semantic search, conversation management, and contextual response generation.