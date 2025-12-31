# Research Summary: Book RAG Chatbot

## Decision: Technology Stack Selection
**Rationale**: Selected Python 3.11 with FastAPI for the backend due to its async capabilities and excellent support for API development. OpenAI Agents/ChatKit SDKs were chosen as specified in the requirements for RAG functionality. Qdrant Cloud was selected for vector storage as specified, and Neon Serverless Postgres for metadata storage.

## Decision: Architecture Pattern
**Rationale**: Chose a web application architecture with a separate backend and frontend to maintain clear separation of concerns. The backend handles all RAG processing, data management, and API endpoints, while the frontend provides the UI integrated into the Docusaurus book platform.

## Decision: Data Storage Approach
**Rationale**: Implemented a dual storage approach with Qdrant Cloud for vector embeddings (enabling semantic search of book content) and Neon Serverless Postgres for metadata and conversation history. This leverages the strengths of both technologies.

## Decision: Retrieval-Augmented Generation Implementation
**Rationale**: Used the OpenAI Agents/ChatKit SDKs to implement RAG functionality as specified. The system will retrieve relevant book content based on user queries and generate responses grounded in that content.

## Decision: Citation System
**Rationale**: Implemented a citation system to track and display the sources of information in responses, ensuring transparency and allowing users to verify information against the original book content.

## Decision: Selected-Text Mode
**Rationale**: Implemented a selected-text mode as specified in requirements, where the chatbot will restrict its answers to only the text selected by the user when this mode is active.