//! Streamed run over prior conversation history for a Rig-based
//! (vectorless) RAG agent.
//!
//! This adapts Rig's `agent_stream_chat` example: it builds an [`Agent`],
//! runs `agent.prompt(prompt).history(history).stream()`, and collects the
//! streamed run's final response. Requires a provider model (for example
//! `OPENAI_API_KEY`); the helpers are generic over any
//! [`DynModel<Completion>`].
//!
//! The two entry points are:
//!
//! - [`stream_chat`] (and its aliases) build an agent from a model, run the
//!   prompt over the supplied history with streaming, and return the final
//!   response text.
//! - [`stream_chat_stream`] (and its aliases) return the raw
//!   [`StreamingResult`] so a caller can observe every
//!   [`MultiTurnStreamItem`] as it is produced.
//!
//! [`collect_stream_final_response`] is the lower-level collector taken
//! straight from the example and works on any [`StreamingResult`].

use rig::prelude::*;

use anyhow::{Result, anyhow};
use futures::StreamExt;

use rig::DynModel;
use rig::operation::Completion;
use rig::agent::PromptResponse;

/// The preamble used by the adapted example.
pub const PREAMBLE: &str = "You are a comedian here to entertain the user using humour and jokes.";

/// The prompt used by the adapted example.
pub const PROMPT: &str = "Entertain me!";

/// Sample prior conversation history, taken from the adapted example.
pub fn sample_history() -> Vec<Message> {
    vec![
        Message::user("Tell me a joke!"),
        Message::assistant("Why did the chicken cross the road?\n\nTo get to the other side!"),
    ]
}

/// Alias for [`sample_history`].
pub fn sample_conversation() -> Vec<Message> {
    sample_history()
}

/// Alias for [`sample_history`].
pub fn example_history() -> Vec<Message> {
    sample_history()
}

/// Alias for [`sample_history`].
pub fn default_history() -> Vec<Message> {
    sample_history()
}

/// Build an [`Agent`] with the example preamble over `model`.
pub fn build_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    AgentBuilder::new(model).preamble(PREAMBLE).build()
}

/// Alias for [`build_agent`].
pub fn agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    build_agent(model)
}

/// Alias for [`build_agent`].
pub fn make_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    build_agent(model)
}

/// Alias for [`build_agent`].
pub fn new_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    build_agent(model)
}

/// Collect the final response text from a [`StreamingResult`].
///
/// This is the helper taken directly from Rig's `agent_stream_chat`
/// example: it drains the stream, capturing the
/// [`MultiTurnStreamItem::FinalResponse`] payload.
pub async fn collect_stream_final_response(stream: &mut StreamingResult) -> Result<String> {
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item? {
            final_response = Some(response.output().to_owned());
        }
    }
    final_response.ok_or_else(|| anyhow!("stream finished without a final response"))
}

/// Alias-style collector that returns the [`PromptResponse`] instead of just
/// its text. Drains `stream` and returns the last `FinalResponse`.
pub async fn collect_stream_final(stream: &mut StreamingResult) -> Result<PromptResponse> {
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item? {
            final_response = Some(response);
        }
    }
    final_response.ok_or_else(|| anyhow!("stream finished without a final response"))
}

/// Alias for [`collect_stream_final`].
pub async fn collect_final_response(stream: &mut StreamingResult) -> Result<PromptResponse> {
    collect_stream_final(stream).await
}

/// Alias for [`collect_stream_final_response`].
pub async fn collect_final_text(stream: &mut StreamingResult) -> Result<String> {
    collect_stream_final_response(stream).await
}

// ---------------------------------------------------------------------------
// Model-based helpers: build an agent from a model, then stream.
// ---------------------------------------------------------------------------

/// Run `prompt` over `history` with streaming, using an agent built from
/// `model`, and return the final response text.
pub async fn stream_chat<M, P, H, T>(model: M, prompt: P, history: H) -> Result<String>
where
    M: Into<DynModel<Completion>>,
    P: Into<Message>,
    H: IntoIterator<Item = T>,
    T: Into<Message>,
{
    let agent = build_agent(model);
    let mut stream = agent.prompt(prompt).history(history).stream();
    collect_stream_final_response(&mut stream).await
}

/// Return the raw [`StreamingResult`] for `prompt` over `history`, using an
/// agent built from `model`. Nothing runs until the stream is polled.
pub fn stream_chat_stream<M, P, H, T>(model: M, prompt: P, history: H) -> StreamingResult
where
    M: Into<DynModel<Completion>>,
    P: Into<Message>,
    H: IntoIterator<Item = T>,
    T: Into<Message>,
{
    let agent = build_agent(model);
    agent.prompt(prompt).history(history).stream()
}

/// Alias-style async helper returning the [`PromptResponse`].
pub async fn stream_chat_response<M, P, H, T>(
    model: M,
    prompt: P,
    history: H,
) -> Result<PromptResponse>
where
    M: Into<DynModel<Completion>>,
    P: Into<Message>,
    H: IntoIterator<Item = T>,
    T: Into<Message>,
{
    let agent = build_agent(model);
    let mut stream = agent.prompt(prompt).history(history).stream();
    collect_stream_final(&mut stream).await
}

// ---------------------------------------------------------------------------
// Agent-based helpers: stream over an existing agent.
// ---------------------------------------------------------------------------

/// Run `prompt` over `history` with streaming using an existing `agent`,
/// returning the final response text.
pub async fn stream_agent_chat<P, H, T>(
    agent: &Agent,
    prompt: P,
    history: H,
) -> Result<String>
where
    P: Into<Message>,
    H: IntoIterator<Item = T>,
    T: Into<Message>,
{
    let mut stream = agent.prompt(prompt).history(history).stream();
    collect_stream_final_response(&mut stream).await
}

/// Return the raw [`StreamingResult`] for `prompt` over `history` using an
/// existing `agent`.
pub fn stream_agent_chat_stream<P, H, T>(
    agent: &Agent,
    prompt: P,
    history: H,
) -> StreamingResult
where
    P: Into<Message>,
    H: IntoIterator<Item = T>,
    T: Into<Message>,
{
    agent.prompt(prompt).history(history).stream()
}

/// Alias-style async helper returning the [`PromptResponse`] for an agent.
pub async fn stream_agent_chat_response<P, H, T>(
    agent: &Agent,
    prompt: P,
    history: H,
) -> Result<PromptResponse>
where
    P: Into<Message>,
    H: IntoIterator<Item = T>,
    T: Into<Message>,
{
    let mut stream = agent.prompt(prompt).history(history).stream();
    collect_stream_final(&mut stream).await
}

// ---------------------------------------------------------------------------
// Aliases (model-based, text result)
// ---------------------------------------------------------------------------

pub use stream_chat as streamed_chat;
pub use stream_chat as stream_chat_text;
pub use stream_chat as run_stream_chat;
pub use stream_chat as run_streamed_chat;
pub use stream_chat as run_streamed;
pub use stream_chat as run_stream;
pub use stream_chat as run_streaming;
pub use stream_chat as run_streaming_chat;
pub use stream_chat as chat_stream_text;
pub use stream_chat as stream_over_history;
pub use stream_chat as streamed_over_history;
pub use stream_chat as run_over_history;
pub use stream_chat as stream_chat_over_history;
pub use stream_chat as streamed_run_over_history;
pub use stream_chat as stream_run;
pub use stream_chat as streamed_run;
pub use stream_chat as chat_stream_response_text;
pub use stream_chat as agent_stream_chat_text;

// ---------------------------------------------------------------------------
// Aliases (model-based, raw stream)
// ---------------------------------------------------------------------------

pub use stream_chat_stream as streamed_chat_stream;
pub use stream_chat_stream as stream_chat_streaming;
pub use stream_chat_stream as start_stream_chat;
pub use stream_chat_stream as open_stream_chat;
pub use stream_chat_stream as stream_over_history_stream;
pub use stream_chat_stream as streamed_over_history_stream;
pub use stream_chat_stream as run_stream_chat_stream;
pub use stream_chat_stream as chat_stream;

// ---------------------------------------------------------------------------
// Aliases (model-based, PromptResponse result)
// ---------------------------------------------------------------------------

pub use stream_chat_response as streamed_chat_response;
pub use stream_chat_response as run_stream_chat_response;
pub use stream_chat_response as stream_chat_prompt_response;

// ---------------------------------------------------------------------------
// Aliases (agent-based, text result)
// ---------------------------------------------------------------------------

pub use stream_agent_chat as agent_stream_chat;
pub use stream_agent_chat as streamed_agent_chat;
pub use stream_agent_chat as run_agent_stream_chat;
pub use stream_agent_chat as run_agent_streamed_chat;
pub use stream_agent_chat as agent_stream;
pub use stream_agent_chat as stream_agent;
pub use stream_agent_chat as chat_stream_agent;

// ---------------------------------------------------------------------------
// Aliases (agent-based, raw stream)
// ---------------------------------------------------------------------------

pub use stream_agent_chat_stream as agent_stream_chat_stream;
pub use stream_agent_chat_stream as streamed_agent_chat_stream;
pub use stream_agent_chat_stream as stream_agent_stream;
pub use stream_agent_chat_stream as agent_stream_stream;

// ---------------------------------------------------------------------------
// Aliases (agent-based, PromptResponse result)
// ---------------------------------------------------------------------------

pub use stream_agent_chat_response as agent_stream_chat_response;
pub use stream_agent_chat_response as streamed_agent_chat_response;

// ---------------------------------------------------------------------------
// OpenAI convenience (mirrors other modules)
// ---------------------------------------------------------------------------

/// Build an agent using an OpenAI client from the environment.
pub fn openai_agent() -> Result<Agent> {
    let client = rig::providers::openai::OpenAI::from_env()?;
    Ok(build_agent(client.completion(rig::providers::openai::GPT_4)))
}

/// Alias for [`openai_agent`].
pub fn openai_streaming_agent() -> Result<Agent> {
    openai_agent()
}

/// Alias for [`openai_agent`].
pub fn openai_chat_agent() -> Result<Agent> {
    openai_agent()
}

/// Run a streamed chat over `history` using an OpenAI client from the
/// environment, returning the final response text.
pub async fn stream_chat_openai<P, H, T>(prompt: P, history: H) -> Result<String>
where
    P: Into<Message>,
    H: IntoIterator<Item = T>,
    T: Into<Message>,
{
    let agent = openai_agent()?;
    stream_agent_chat(&agent, prompt, history).await
}

/// Alias for [`stream_chat_openai`].
pub async fn streamed_chat_openai<P, H, T>(prompt: P, history: H) -> Result<String>
where
    P: Into<Message>,
    H: IntoIterator<Item = T>,
    T: Into<Message>,
{
    stream_chat_openai(prompt, history).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_history_has_two_messages() {
        assert_eq!(sample_history().len(), 2);
    }

    #[test]
    fn constants_are_non_empty() {
        assert!(!PREAMBLE.is_empty());
        assert!(!PROMPT.is_empty());
    }

    #[test]
    fn openai_constant_exists() {
        let _ = rig::providers::openai::GPT_4;
    }
}
