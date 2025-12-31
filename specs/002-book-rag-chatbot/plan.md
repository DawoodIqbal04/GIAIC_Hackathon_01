# Implementation Plan: Book RAG Chatbot

**Branch**: `002-book-rag-chatbot` | **Date**: 2025-12-31 | **Spec**: [specs/002-book-rag-chatbot/spec.md](specs/002-book-rag-chatbot/spec.md)
**Input**: Feature specification from `/specs/002-book-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build and embed a Retrieval-Augmented Generation (RAG) chatbot within the published book that can answer user questions about the book's content. The system will use OpenAI Agents/ChatKit SDKs with FastAPI backend, Neon Serverless Postgres for metadata, and Qdrant Cloud for vector storage. The chatbot will retrieve relevant information from the book content and generate accurate responses while ensuring all answers are grounded in the book content without hallucinating information. The system will also support selected-text mode where answers are restricted to user-selected text.

## Technical Context

**Language/Version**: Python 3.11 (for FastAPI backend and RAG processing)
**Primary Dependencies**:
- FastAPI (web framework)
- OpenAI Agents/ChatKit SDKs (for RAG functionality)
- Qdrant (vector database for document retrieval)
- Neon Serverless Postgres (metadata storage)
- Docusaurus (book platform)
**Storage**:
- Vector storage: Qdrant Cloud (Free Tier) for document embeddings
- Metadata storage: Neon Serverless Postgres
- Source of truth: Docusaurus Markdown/MDX book content
**Testing**: pytest (for backend API and RAG functionality)
**Target Platform**: Web application integrated with Docusaurus book platform
**Project Type**: Web application with backend API and RAG processing
**Performance Goals**:
- Response time: Under 10 seconds for query processing (as specified in success criteria)
- Concurrency: Support multiple simultaneous users
**Constraints**:
- Responses must be grounded in book content only (no hallucinations)
- Selected-text mode: Answers restricted to user-selected text when in that mode
- Free tier limitations: Qdrant Cloud Free Tier and Neon Serverless Postgres
**Scale/Scope**: Single book RAG chatbot system with multi-turn conversation support

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Based on the Embedded Retrieval-Augmented Generation (RAG) Chatbot Constitution:
- All chatbot answers must be derived strictly from indexed book content or user-selected text
- Retrieval process precedes response generation with no free-form answering
- Full traceability maintained between user queries, retrieved sources, and final answers
- All content follows spec-driven development methodology using Spec-Kit Plus
- Traceability maintained between book content, specifications, and chatbot responses
- Modularity achieved while keeping components synchronized
- All responses must be grounded in indexed book content or user-selected text
- No hallucinated facts or external knowledge beyond the book corpus
- Responses must cite retrieved sections or chapter references

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── query.py          # Query entity and validation
│   │   ├── response.py       # Response entity and validation
│   │   ├── document.py       # Document entity for book content
│   │   └── citation.py       # Citation entity for source references
│   ├── services/
│   │   ├── rag_service.py    # Core RAG functionality
│   │   ├── retrieval_service.py # Document retrieval logic
│   │   ├── generation_service.py # Response generation logic
│   │   └── citation_service.py # Citation generation logic
│   ├── api/
│   │   ├── main.py           # FastAPI app entry point
│   │   ├── routes/
│   │   │   ├── chat.py       # Chat endpoint
│   │   │   ├── query.py      # Query processing endpoint
│   │   │   └── health.py     # Health check endpoint
│   │   └── dependencies.py   # API dependencies
│   ├── config/
│   │   └── settings.py       # Application settings
│   └── utils/
│       ├── validators.py     # Input validation utilities
│       └── helpers.py        # General helper functions
└── tests/
    ├── unit/
    │   ├── test_models/
    │   ├── test_services/
    │   └── test_api/
    ├── integration/
    │   └── test_endpoints/
    └── contract/
        └── test_api_contracts/

website/
├── src/
│   ├── components/
│   │   └── RagChatbot/
│   │       ├── ChatInterface.jsx  # Main chat UI component
│   │       ├── Message.jsx        # Individual message display
│   │       ├── QueryInput.jsx     # Query input field
│   │       └── Citations.jsx      # Citations display
│   └── pages/
│       └── ChatPage.jsx           # Page integrating chatbot
├── static/
│   └── js/
│       └── rag-chatbot-integration.js  # Integration with Docusaurus
└── docusaurus.config.js             # Docusaurus configuration
```

**Structure Decision**: Web application with separate backend (FastAPI) and frontend (Docusaurus React components) to support the RAG chatbot functionality. The backend handles all RAG processing, while the frontend provides the UI integrated into the Docusaurus book platform.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
