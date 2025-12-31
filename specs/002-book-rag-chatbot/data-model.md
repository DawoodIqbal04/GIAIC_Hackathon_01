# Data Model: Book RAG Chatbot

## Entities

### Query
- **id**: UUID (Primary Key)
- **content**: String (The user's natural language query)
- **timestamp**: DateTime (When the query was submitted)
- **user_id**: UUID (Optional, for tracking user sessions)
- **conversation_id**: UUID (To maintain conversation context)
- **selected_text**: String (Optional, text selected by user for selected-text mode)

### Document
- **id**: UUID (Primary Key)
- **title**: String (Title of the document/chapter/section)
- **content**: Text (The actual content of the document)
- **source_path**: String (Path to the original document in the book)
- **embedding**: Vector (Vector representation for semantic search)
- **metadata**: JSON (Additional metadata like author, date, etc.)
- **created_at**: DateTime
- **updated_at**: DateTime

### RetrievedContext
- **id**: UUID (Primary Key)
- **query_id**: UUID (Foreign Key to Query)
- **document_id**: UUID (Foreign Key to Document)
- **content**: Text (The portion of the document retrieved)
- **relevance_score**: Float (Score indicating relevance to the query)
- **source_citation**: String (Reference to the original location in the book)

### GeneratedResponse
- **id**: UUID (Primary Key)
- **query_id**: UUID (Foreign Key to Query)
- **content**: Text (The generated response)
- **timestamp**: DateTime
- **citations**: JSON (List of citations used in the response)
- **conversation_context**: JSON (Context from previous interactions)

### Citation
- **id**: UUID (Primary Key)
- **response_id**: UUID (Foreign Key to GeneratedResponse)
- **document_id**: UUID (Foreign Key to Document)
- **source_reference**: String (Specific reference like chapter, section, page)
- **content_snippet**: Text (The relevant snippet from the source)

### Conversation
- **id**: UUID (Primary Key)
- **user_id**: UUID (Optional, for registered users)
- **session_id**: String (For anonymous sessions)
- **created_at**: DateTime
- **updated_at**: DateTime
- **metadata**: JSON (Additional conversation metadata)

## Relationships

- Query → GeneratedResponse (One-to-One)
- Query → RetrievedContext (One-to-Many)
- GeneratedResponse → Citation (One-to-Many)
- RetrievedContext → Document (Many-to-One)
- Citation → Document (Many-to-One)
- Conversation → Query (One-to-Many)
- Conversation → GeneratedResponse (One-to-Many)

## Validation Rules

- Query.content must be between 10 and 5000 characters
- Document.content must not exceed 10,000 characters per chunk
- Relevance_score must be between 0 and 1
- Query must have either content or selected_text (or both)
- Conversation must be active for new queries to be added

## State Transitions

- Query: PENDING → PROCESSING → COMPLETED
- RetrievedContext: CREATED → RANKED → SELECTED
- GeneratedResponse: DRAFT → GENERATED → REVIEWED (if needed) → PUBLISHED