# RAG Chatbot API Contract

## Base URL
`https://api.yourbook.com/v1`

## Endpoints

### POST /chat/query
Submit a query to the RAG chatbot and receive a response.

#### Request
```json
{
  "query": "What are the main principles of machine learning?",
  "conversation_id": "uuid-string",
  "selected_text": "Optional text selected by user",
  "include_citations": true
}
```

#### Response (200 OK)
```json
{
  "response_id": "uuid-string",
  "content": "The main principles of machine learning include supervised learning, unsupervised learning, and reinforcement learning...",
  "citations": [
    {
      "source_reference": "Chapter 3, Section 2",
      "content_snippet": "Machine learning algorithms can be categorized into several types...",
      "document_id": "uuid-string"
    }
  ],
  "conversation_id": "uuid-string",
  "timestamp": "2025-12-31T10:00:00Z"
}
```

#### Error Responses
- 400: Invalid request format
- 429: Rate limit exceeded
- 500: Internal server error

### POST /chat/conversation/start
Start a new conversation.

#### Request
```json
{
  "user_id": "optional-uuid-string"
}
```

#### Response (201 Created)
```json
{
  "conversation_id": "uuid-string",
  "timestamp": "2025-12-31T10:00:00Z"
}
```

### GET /chat/conversation/{conversation_id}
Get conversation history.

#### Response (200 OK)
```json
{
  "conversation_id": "uuid-string",
  "messages": [
    {
      "type": "query|response",
      "content": "string",
      "timestamp": "2025-12-31T10:00:00Z",
      "citations": ["array of citation objects if type is response"]
    }
  ]
}
```

### POST /documents/index
Index book content for retrieval.

#### Request
```json
{
  "source_path": "/path/to/book/content",
  "title": "Chapter Title",
  "metadata": {
    "author": "Author Name",
    "section": "Section Number",
    "page_range": "10-25"
  }
}
```

#### Response (202 Accepted)
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "estimated_completion": "2025-12-31T10:05:00Z"
}
```

### GET /documents/index/status/{job_id}
Check indexing job status.

#### Response (200 OK)
```json
{
  "job_id": "uuid-string",
  "status": "completed|processing|failed",
  "progress": 0.75,
  "details": "Optional details about the indexing process"
}
```

### GET /health
Health check endpoint.

#### Response (200 OK)
```json
{
  "status": "healthy",
  "timestamp": "2025-12-31T10:00:00Z",
  "dependencies": {
    "qdrant": "connected",
    "postgres": "connected",
    "openai": "connected"
  }
}
```