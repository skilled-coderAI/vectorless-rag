//! Hand-driven agent loop helpers for a Rig-based (vectorless) RAG agent.
//!
//! This module wraps Rig's sans-I/O [`AgentRun`] state machine and exposes
//! [`drive_turns`], a helper that drives a run turn by turn: it issues the
//! raw model call, resolves invalid tool calls, executes the pending tools,
//! and returns the final [`PromptResponse`].
//!
//! Because [`AgentRun`] is fully serializable between steps, the helper also
//! demonstrates pausing a run while tool calls are pending and resuming it
//! from its serialized form before executing those calls.
//!
//! Requires a provider model (for example `OPENAI_API_KEY`).

use std::collections::BTreeSet;

use anyhow::{Context, Result};
use rig::DynModel;
use rig::agent::InvalidToolCallAction;
use rig::completion::CompletionRequest;
use rig::message::Message;
use rig::operation::Completion;
use rig::run::{
    AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome, PendingToolCall, PromptResponse, TurnPolicy,
};
use rig::tool::{ToolContext, ToolSet};

/// Build a fresh [`AgentRun`] for `prompt` with the given model-call budget.
pub fn new_run(prompt: impl Into<Message>, max_turns: usize) -> AgentRun {
    AgentRun::new(prompt).max_turns(max_turns)
}

/// Drive `run` to completion using `model` and the tools registered in `tools`.
///
/// `preamble` is sent with every model call. Returns the run's final response.
pub async fn drive_turns(
    model: &DynModel<Completion>,
    prompt: impl Into<Message>,
    preamble: impl Into<String>,
    tools: &ToolSet,
    max_turns: usize,
) -> Result<PromptResponse> {
    let preamble = preamble.into();
    let tool_definitions = tools.tool_definitions();
    let tool_names: BTreeSet<String> = tool_definitions
        .iter()
        .map(|definition| definition.name.to_string())
        .collect();

    let mut run = AgentRun::new(prompt).max_turns(max_turns);

    loop {
        match run.next_step()? {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => {
                println!("-> model call #{turn}");

                let response = model
                    .call(
                        CompletionRequest::new(prompt)
                            .messages(history)
                            .preamble(preamble.clone())
                            .tools(tool_definitions.clone()),
                    )
                    .await
                    .context("model call failed")?;

                let mut outcome = run.model_response(ModelTurn::new(
                    response.head(),
                    response.choice.clone(),
                    response.usage,
                    TurnPolicy::new(tool_names.clone(), None, None)
                        .context("invalid turn policy")?,
                    response.raw.clone(),
                ))?;

                while let ModelTurnOutcome::NeedsResolution(context) = outcome {
                    eprintln!("model called unknown tool `{}`", context.tool_name);
                    outcome = run
                        .resolve_invalid_tool_call(InvalidToolCallAction::fail())
                        .context("failed to resolve invalid tool call")?;
                }
            }
            AgentRunStep::CallTools { .. } => {
                // The whole run is serializable while tool calls are pending:
                // persist it, then resume from that serialized form before
                // executing the (re-emitted) pending calls.
                let suspended = serde_json::to_string(&run).context("failed to serialize run")?;
                let mut resumed: AgentRun =
                    serde_json::from_str(&suspended).context("failed to resume run")?;

                let AgentRunStep::CallTools { calls } = resumed.next_step()? else {
                    anyhow::bail!("resumed run did not re-emit its pending tool calls");
                };

                for call in calls {
                    match call {
                        PendingToolCall::Execute(call) => {
                            let args = call.arguments().to_string();
                            println!("-> executing {}({args})", call.name());

                            let mut context = ToolContext::new();
                            let result = tools
                                .execute(call.name(), args, &mut context)
                                .await;
                            resumed
                                .answer(call.answer(result))
                                .context("failed to answer tool call")?;
                        }
                        PendingToolCall::Malformed(call) => {
                            resumed
                                .answer(call.answer(None))
                                .context("failed to answer malformed tool call")?;
                        }
                        _ => anyhow::bail!("unsupported pending tool call"),
                    }
                }

                run = resumed;
            }
            AgentRunStep::Done(response) => {
                println!(
                    "done: {} ({} model call(s))",
                    response.output(),
                    response.completion_calls.len()
                );
                return Ok(response);
            }
        }
    }
}

/// Alias for [`drive_turns`].
pub async fn drive_agent_run(
    model: &DynModel<Completion>,
    prompt: impl Into<Message>,
    preamble: impl Into<String>,
    tools: &ToolSet,
    max_turns: usize,
) -> Result<PromptResponse> {
    drive_turns(model, prompt, preamble, tools, max_turns).await
}

/// Alias for [`drive_turns`].
pub async fn hand_driven_run(
    model: &DynModel<Completion>,
    prompt: impl Into<Message>,
    preamble: impl Into<String>,
    tools: &ToolSet,
    max_turns: usize,
) -> Result<PromptResponse> {
    drive_turns(model, prompt, preamble, tools, max_turns).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_run_emits_model_call() {
        let mut run = new_run("What is 2 + 2?", 2);
        assert!(matches!(
            run.next_step().unwrap(),
            AgentRunStep::CallModel { turn: 1, .. }
        ));
    }

    #[test]
    fn run_is_serializable_before_start() {
        let run = new_run("What is 2 + 2?", 2);
        let serialized = serde_json::to_string(&run).unwrap();
        let restored: AgentRun = serde_json::from_str(&serialized).unwrap();
        assert_eq!(restored.turn(), 0);
    }
}
