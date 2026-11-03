# Vectorless RAG Agent - Progress Tracker

Status: In progress
Framework: Rig (0.46.0)
Style: Vectorless retrieval (no embeddings, no vector store)

## Phase 0 - Foundation
- [x] Import `agent_tool_call_streaming` example into project.
- [x] Convert to standalone crate (`vectorless-rag`).
- [x] Verify `cargo build` succeeds.
- [x] Commit initial import (conventional commit).

## Phase 1 - Agent core and tools
- [x] Add `agent` minimal setup.
- [x] Add `multi_turn_agent`.
- [x] Add tool registration (`agent_with_tools`).
- [x] Add manual tool calls (`manual_tool_calls`).
- [x] Add forced first-turn tool use (`force_tool_first_turn`).

## Phase 2 - Documents and context (no vectors)
- [x] Add document handling (`documents`).
- [x] Add PDF handling (`pdf_agent`).
- [x] Add file loaders (`agent_with_loaders`).
- [x] Add small context documents (`agent_with_context`).

## Phase 3 - Vectorless retrieval
- [x] Add retrieval pipeline (`chain` with `parallel!` and `lookup`).
- [x] Add typed extraction (`extractor`).

## Phase 4 - Memory and state
- [x] Add in-memory conversation memory (`agent_with_memory`).
- [x] Add hand-driven loop (`agent_run_stepping`).

## Phase 5 - Streaming and reliability
- [x] Add streamed run over history (`agent_stream_chat`).
- [x] Add request/response/tool hooks (`request_hook`).
- [x] Add retry hook (`agent_with_retry_hook`).
- [x] Add tool failure handling (`tool_result_outcomes`).

## Phase 6 - Polish
- [ ] Add README with run instructions.
- [x] Add basic tests.
- [ ] Final review and conventional commit.

## Notes
- Vector-based examples are intentionally excluded: `rag`, `vector_search`,
  `vector_search_cohere`, `custom_vector_store`, `rag_dynamic_tools`.
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.