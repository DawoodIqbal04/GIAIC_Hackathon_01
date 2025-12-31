---
id: 10
title: Book RAG Chatbot Plan
stage: plan
date: 2025-12-31
surface: agent
model: Qwen
feature: book-rag-chatbot
branch: 002-book-rag-chatbot
user: User
command: /sp.plan
labels: ["plan", "rag", "chatbot", "book", "architecture"]
links:
  spec: specs/002-book-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/002-book-rag-chatbot/plan.md
 - specs/002-book-rag-chatbot/research.md
 - specs/002-book-rag-chatbot/data-model.md
 - specs/002-book-rag-chatbot/quickstart.md
 - specs/002-book-rag-chatbot/contracts/api-contract.md
 - QWEN.md
tests:
 - null
---

## Prompt

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. **Setup**: Run `.specify/scripts/powershell/setup-plan.ps1 -Json` from repo root and parse JSON for FEATURE_SPEC, IMPL_PLAN, SPECS_DIR, BRANCH. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Load context**: Read FEATURE_SPEC and `.specify/memory/constitution.md`. Load IMPL_PLAN template (already copied).

3. **Execute plan workflow**: Follow the structure in IMPL_PLAN template to:
   - Fill Technical Context (mark unknowns as "NEEDS CLARIFICATION")
   - Fill Constitution Check section from constitution
   - Evaluate gates (ERROR if violations unjustified)
   - Phase 0: Generate research.md (resolve all NEEDS CLARIFICATION)
   - Phase 1: Generate data-model.md, contracts/, quickstart.md
   - Phase 1: Update agent context by running the agent script
   - Re-evaluate Constitution Check post-design

4. **Stop and report**: Command ends after Phase 2 planning. Report branch, IMPL_PLAN path, and generated artifacts.

## Phases

### Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:

   ```text
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

### Phase 1: Design & Contracts

**Prerequisites:** `research.md` complete

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Agent context update**:
   - Run `.specify/scripts/powershell/update-agent-context.ps1 -AgentType qwen`
   - These scripts detect which AI agent is in use
   - Update the appropriate agent-specific context file
   - Add only new technology from current plan
   - Preserve manual additions between markers

**Output**: data-model.md, /contracts/*, quickstart.md, agent-specific file

## Key rules

- Use absolute paths
- ERROR on gate failures or unresolved clarifications

---

As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

1) Determine Stage
   - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

2) Generate Title and Determine Routing:
   - Generate Title: 3–7 words (slug for filename)
   - Route is automatically determined by stage:
     - `constitution` → `history/prompts/constitution/`
     - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
     - `general` → `history/prompts/general/`

3) Create and Fill PHR (Shell first; fallback agent‑native)
   - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
   - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
   - If the script fails:
     - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
     - Allocate an ID; compute the output path based on stage from step 2; write the file
     - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

4) Validate + report
   - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
   - On failure: warn, don't block. Skip only for `/sp.phr`.


/sp.plan Integrated RAG Chatbot Development: Build and embed a Retrieval-Augmented Generation (RAG) chatbot within the published book. This chatbot, utilizing the OpenAI Agents/ChatKit SDKs, FastAPI, Neon Serverless Postgres database, and Qdrant Cloud Free Tier, must be able to answer user questions about the book's content, including answering questions based only on text selected by the user.

## Response snapshot

The implementation planning for the Book RAG Chatbot has been successfully completed with the following artifacts generated:

**Branch Name**: `002-book-rag-chatbot`
**Implementation Plan Path**: `specs/002-book-rag-chatbot/plan.md`

**Generated Artifacts**:
1. **Plan**: `specs/002-book-rag-chatbot/plan.md` - Complete implementation plan with technical context, project structure, and constitution check
2. **Research**: `specs/002-book-rag-chatbot/research.md` - Research summary with technology decisions and rationale
3. **Data Model**: `specs/002-book-rag-chatbot/data-model.md` - Complete data model with entities, relationships, and validation rules
4. **Quickstart Guide**: `specs/002-book-rag-chatbot/quickstart.md` - Setup and usage guide
5. **API Contracts**: `specs/002-book-rag-chatbot/contracts/api-contract.md` - OpenAPI specification for the RAG chatbot API
6. **Agent Context**: Updated `QWEN.md` with project-specific context

All artifacts have been created and the implementation plan is complete, meeting all constitutional requirements for the RAG chatbot project.

## Outcome

- ✅ Impact: Created comprehensive implementation plan for the Book RAG Chatbot with all required artifacts
- 🧪 Tests: No tests required for planning phase
- 📁 Files: Generated plan.md, research.md, data-model.md, quickstart.md, contracts/api-contract.md, and updated QWEN.md
- 🔁 Next prompts: Create tasks using /sp.tasks to break the plan into actionable items
- 🧠 Reflection: The planning process successfully created a complete architecture for the RAG chatbot that meets all constitutional requirements while using the specified technologies.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Create tasks from the implementation plan