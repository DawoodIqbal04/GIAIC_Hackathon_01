<!--
Sync Impact Report:
Version change: 1.1.0 → 1.2.0
Modified principles: I, II, III (updated to reflect RAG chatbot project requirements)
Added sections: None (revised existing principles to align with new project)
Removed sections: Physical AI and Humanoid Robotics specific content
Templates requiring updates:
- .specify/templates/plan-template.md ✅ updated
- .specify/templates/spec-template.md ⚠ pending (no direct references to update)
- .specify/templates/tasks-template.md ⚠ pending (no direct references to update)
- README.md ⚠ pending (no direct references to update)
Follow-up TODOs: None
-->

# Embedded Retrieval-Augmented Generation (RAG) Chatbot Constitution

## Core Principles

### I. Grounded Responses
All chatbot answers must be derived strictly from indexed book content or explicitly user-selected text; No hallucinated facts or external knowledge beyond the book corpus; Responses must cite retrieved sections or chapter references.

### II. Deterministic Retrieval Before Generation
Retrieval process precedes response generation with no free-form answering; Clear separation maintained between retrieval, reasoning, and response generation; Fail gracefully when relevant context is unavailable.

### III. Traceability and Auditability
Full traceability maintained between user queries, retrieved sources, and final answers; System behavior is reproducible and auditable; Internal QA confirms zero unsupported or out-of-scope responses.

### IV. Spec-Driven Development with Spec-Kit Plus
All content and system components follow spec-driven development methodology using Spec-Kit Plus; All deliverables begin with clear specifications before implementation; Traceability maintained between requirements, implementation, and verification.

### V. Content-System Traceability
Full traceability maintained between book content, specifications, and chatbot responses; Changes in one component are reflected appropriately in others; Synchronization mechanisms ensure consistency across all system parts.

### VI. Modularity with Synchronized Components
Book chapters, specifications, and chatbot knowledge remain decoupled yet synchronized; Modular design enables independent evolution of components while maintaining coherence; System interfaces clearly defined to support modularity.

## Content Standards and Constraints

- All responses must be grounded in indexed book content or user-selected text
- Platform: Docusaurus-based technical book with integrated RAG chatbot, deployed via GitHub Pages
- Retrieval source of truth: Docusaurus Markdown/MDX content
- Vector storage: Qdrant Cloud (Free Tier)
- Metadata storage: Neon Serverless Postgres
- API layer: FastAPI
- Agent framework: OpenAI Agents / ChatKit SDKs
- Selected-text mode must restrict answers exclusively to the provided selection
- No training or fine-tuning on external datasets
- Clear distinction between retrieved content and generated responses
- Accessibility considerations for technical readers with diverse backgrounds

## Development Workflow and Quality Standards

- Technical accuracy verified through authoritative book sources
- Content undergoes technical review by subject matter experts
- System components tested for accuracy and relevance to book content
- Continuous integration ensures content integrity and system functionality
- Regular audits verify alignment between book content and chatbot knowledge base
- Version control maintains history of content changes and their rationale
- Chatbot consistently answers questions using only retrieved content
- Selected-text queries are strictly scoped and enforced
- Chatbot integrates seamlessly within the published book UI

## Governance

This constitution governs all aspects of the Embedded Retrieval-Augmented Generation (RAG) Chatbot project. All contributions must comply with these principles. Changes to this constitution require explicit approval and documentation of the rationale. All pull requests and reviews must verify constitutional compliance before merging.

**Version**: 1.2.0 | **Ratified**: 2025-06-13 | **Last Amended**: 2025-12-31
