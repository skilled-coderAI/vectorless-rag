# vectorless-rag

A vectorless RAG agent built with the [Rig](https://github.com/0xPlaygrounds/rig) framework.

This project does not use embeddings or a vector store. It retrieves and
uses documents through lexical search, Rig agents, tools, and conversation
memory.

## Build

```sh
cargo build
```

## Test

```sh
cargo test --lib
```

## Run the original streaming example

```sh
cargo run -- anthropic
cargo run -- openai --model gpt-5.4-mini
```

Provider API keys are read from the environment (for example
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## Modules

- `documents` - document model, text/PDF loading, context helpers.
- `retrieval` - lexical retriever and context formatting.
- `memory` - in-memory conversation memory helpers.
- `extractor` - typed extraction helpers.
- `stepping` - hand-driven agent loop helpers.
- `hooks` - lifecycle, retry, and tool-failure hooks.
- `streaming_chat` - streamed runs over conversation history.

## Progress

See `PROGRESS.md`.