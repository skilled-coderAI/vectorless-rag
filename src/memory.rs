//! Conversation-memory helpers for a Rig-based RAG agent.
//!
//! This wraps Rig's in-memory conversation memory ([`MemoryStore`]) and exposes
//! helpers to either build a memory-backed [`Agent`] or to attach an in-memory
//! store to an existing [`AgentBuilder`].
//!
//! Requires the `memory` feature of `rig` (enabled in this crate's
//! `Cargo.toml`).

use rig::DynModel;
use rig::agent::{Agent, AgentBuilder};
use rig::conversation::Conversation;
use rig::memory::{MemoryStore, SessionError, load, load_async, save, save_async};
use rig::operation::Completion;

/// An in-memory conversation store (Rig's `MemoryStore`).
pub type InMemoryStore = MemoryStore;
/// Alias kept for callers that think in terms of "conversation memory".
pub type InMemoryConversationMemory = MemoryStore;
/// Generic alias for the memory backing a conversation.
pub type ConversationMemory = MemoryStore;

/// Create a fresh in-memory conversation store.
pub fn in_memory_store() -> MemoryStore {
    MemoryStore::default()
}

/// Create a fresh in-memory conversation store (alternative name).
pub fn new_in_memory_store() -> MemoryStore {
    MemoryStore::default()
}

/// Create a fresh in-memory conversation store (alternative name).
pub fn memory_store() -> MemoryStore {
    MemoryStore::default()
}

/// Create a fresh in-memory conversation store (alternative name).
pub fn default_memory_store() -> MemoryStore {
    MemoryStore::default()
}

/// Create a fresh in-memory conversation store (alternative name).
pub fn new_memory() -> MemoryStore {
    MemoryStore::default()
}

/// Create a fresh in-memory conversation store (alternative name).
pub fn in_memory() -> MemoryStore {
    MemoryStore::default()
}

// ---------------------------------------------------------------------------
// Attaching memory to an AgentBuilder
// ---------------------------------------------------------------------------

/// Attach an in-memory conversation store to an [`AgentBuilder`].
pub fn attach_memory(builder: AgentBuilder, store: MemoryStore) -> AgentBuilder {
    builder.memory(store)
}

/// Attach an in-memory conversation store to an [`AgentBuilder`] (alias).
pub fn with_memory(builder: AgentBuilder, store: MemoryStore) -> AgentBuilder {
    builder.memory(store)
}

/// Attach an in-memory conversation store to an [`AgentBuilder`] (alias).
pub fn add_memory(builder: AgentBuilder, store: MemoryStore) -> AgentBuilder {
    builder.memory(store)
}

/// Attach an in-memory conversation store to an [`AgentBuilder`] (alias).
pub fn use_memory(builder: AgentBuilder, store: MemoryStore) -> AgentBuilder {
    builder.memory(store)
}

/// Attach an in-memory conversation store to an [`AgentBuilder`] (alias).
pub fn memory(builder: AgentBuilder, store: MemoryStore) -> AgentBuilder {
    builder.memory(store)
}

/// Attach a default in-memory conversation store to an [`AgentBuilder`].
pub fn attach_in_memory(builder: AgentBuilder) -> AgentBuilder {
    builder.memory(MemoryStore::default())
}

/// Attach a default in-memory conversation store to an [`AgentBuilder`] (alias).
pub fn with_in_memory(builder: AgentBuilder) -> AgentBuilder {
    builder.memory(MemoryStore::default())
}

/// Attach a default in-memory conversation store to an [`AgentBuilder`] (alias).
pub fn with_default_memory(builder: AgentBuilder) -> AgentBuilder {
    builder.memory(MemoryStore::default())
}

/// Attach an in-memory store and name the conversation log it uses.
pub fn attach_memory_with_log<S>(builder: AgentBuilder, store: MemoryStore, log: S) -> AgentBuilder
where
    S: Into<String>,
{
    builder.memory(store).conversation(log)
}

/// Attach an in-memory store and name the conversation log it uses (alias).
pub fn with_memory_and_conversation<S>(
    builder: AgentBuilder,
    store: MemoryStore,
    log: S,
) -> AgentBuilder
where
    S: Into<String>,
{
    builder.memory(store).conversation(log)
}

/// Attach an in-memory store and name the conversation log it uses (alias).
pub fn attach_memory_with_conversation<S>(
    builder: AgentBuilder,
    store: MemoryStore,
    log: S,
) -> AgentBuilder
where
    S: Into<String>,
{
    builder.memory(store).conversation(log)
}

// ---------------------------------------------------------------------------
// Building memory-backed agents
// ---------------------------------------------------------------------------

/// Build an [`Agent`] backed by a default in-memory conversation store.
pub fn memory_backed_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    memory_backed_agent_with_store(model, MemoryStore::default())
}

/// Build an [`Agent`] backed by the given in-memory conversation store.
pub fn memory_backed_agent_with_store<M>(model: M, store: MemoryStore) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    AgentBuilder::new(model).memory(store).build()
}

/// Build an [`Agent`] backed by an in-memory store and a named conversation log.
pub fn memory_backed_agent_with_log<M, S>(model: M, store: MemoryStore, log: S) -> Agent
where
    M: Into<DynModel<Completion>>,
    S: Into<String>,
{
    AgentBuilder::new(model)
        .memory(store)
        .conversation(log)
        .build()
}

/// Build an [`Agent`] backed by a default in-memory store (alias).
pub fn agent_with_memory<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    memory_backed_agent(model)
}

/// Build an [`Agent`] backed by a default in-memory store (alias).
pub fn create_memory_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    memory_backed_agent(model)
}

/// Build an [`Agent`] backed by a default in-memory store (alias).
pub fn create_memory_backed_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    memory_backed_agent(model)
}

/// Build an [`Agent`] backed by a default in-memory store (alias).
pub fn new_memory_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    memory_backed_agent(model)
}

/// Build an [`Agent`] backed by a default in-memory store (alias).
pub fn in_memory_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    memory_backed_agent(model)
}

// ---------------------------------------------------------------------------
// Conversation load / save helpers
// ---------------------------------------------------------------------------

/// Load a conversation from an in-memory store.
pub fn load_conversation(store: &MemoryStore, log: &str) -> Result<Conversation, SessionError> {
    load(store, log)
}

/// Save a conversation to an in-memory store.
pub fn save_conversation(
    store: &MemoryStore,
    log: &str,
    conversation: &mut Conversation,
) -> Result<(), SessionError> {
    save(store, log, conversation)
}

/// Load a conversation from an in-memory store asynchronously.
pub async fn load_conversation_async(
    store: &MemoryStore,
    log: &str,
) -> Result<Conversation, SessionError> {
    load_async(store, log).await
}

/// Save a conversation to an in-memory store asynchronously.
pub async fn save_conversation_async(
    store: &MemoryStore,
    log: &str,
    conversation: &mut Conversation,
) -> Result<(), SessionError> {
    save_async(store, log, conversation).await
}

// Re-export the traits so callers can name them through this module.
pub use rig::memory::{AsyncStore as InMemoryAsyncStore, Store as InMemoryStoreTrait};
