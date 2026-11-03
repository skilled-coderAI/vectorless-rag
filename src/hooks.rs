//! Rig hook patterns for the vectorless RAG agent.
//!
//! Combines three Rig examples into one reusable module:
//!
//! 1. [`LoggingHook`] — a request/response/tool lifecycle logging hook built on
//!    the composing `AgentHook` model (request_hook).
//! 2. [`RetryOnMarker`] — a retry hook with corrective feedback and a per-run
//!    limit, owned in the run-scoped scratchpad (agent_with_retry_hook).
//! 3. Tool failure outcome handling helpers — classify tool failures as
//!    structured facts and apply policy in hooks (tool_result_outcomes).
//!
//! All items compile against `rig` 0.46.0.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, ModelTurnAction,
    ModelTurnFinished, OutcomeAction, OutcomeEvent, RequestPatch,
};
use rig::completion::{Document, Message};
use rig::message::{ToolChoice, UserContent};
use rig::tool::{
    Tool, ToolContext, ToolErrorKind, ToolExecutionError, ToolResult, tool_name,
};

// ---------------------------------------------------------------------------
// 1. Request/response/tool lifecycle logging hook (request_hook)
// ---------------------------------------------------------------------------

/// Observe-only hook that logs the prompt sent on every completion call and the
/// response received on every outcome. Registered first so a later terminating
/// hook can never hide events from it.
#[derive(Clone, Default)]
pub struct LoggingHook;

impl AgentHook for LoggingHook {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if let Message::User { content } = event.prompt {
            let prompt_text = content
                .iter()
                .filter_map(|c| match c {
                    UserContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            if !prompt_text.is_empty() {
                println!(
                    "[run {} · turn {}] sending prompt: {}",
                    ctx.run_id(),
                    ctx.turn(),
                    prompt_text
                );
            }
        }
        CompletionCallAction::continue_run()
    }

    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        // `on_outcome` fires for every effect family; only report completions.
        if let Some(response) = event.completion() {
            println!(
                "[run {}] received response (usage: {:?}, response_id: {:?}): {:?}",
                ctx.run_id(),
                response.usage,
                response.response_id(),
                response.choice
            );
        }
        // Also surface tool results so the full tool lifecycle is observable.
        if let (Some(tool_name), Some(result)) = (event.tool_name(), event.tool_result()) {
            if let Some(error) = result.error() {
                println!(
                    "[run {}] tool `{}` failed: {}",
                    ctx.run_id(),
                    tool_name,
                    error
                );
            } else {
                println!("[run {}] tool `{}` succeeded", ctx.run_id(), tool_name);
            }
        }
        OutcomeAction::proceed()
    }
}

/// Injects an extra context document for the turn via `RequestPatch::extra_context`
/// (passive RAG). Stacked with [`LoggingHook`] and [`SamplingHook`].
#[derive(Clone)]
pub struct ContextHook;

impl AgentHook for ContextHook {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let doc = Document {
            id: "style-guide".to_string(),
            text: "House style: keep jokes short and family-friendly.".to_string(),
            additional_props: Default::default(),
        };
        CompletionCallAction::patch(RequestPatch::new().context(doc))
    }
}

/// Lowers the sampling temperature for the turn via `RequestPatch::temperature`.
/// Its patch merges with [`ContextHook`]'s rather than replacing it.
#[derive(Clone)]
pub struct SamplingHook;

impl AgentHook for SamplingHook {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        CompletionCallAction::patch(RequestPatch::new().temperature(0.2))
    }
}

/// Counts completion calls using the run-scoped [`HookContext`] scratchpad.
#[derive(Clone, Default)]
pub struct TurnCount(pub usize);

#[derive(Clone)]
pub struct TurnCounterHook;

impl AgentHook for TurnCounterHook {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let n = ctx.scratchpad().update(|c: &mut TurnCount| {
            c.0 += 1;
            c.0
        });
        println!("[turn-counter] completion call #{n} this run");
        CompletionCallAction::continue_run()
    }
}

// Aliases for callers that think in terms of the lifecycle hook.
pub type LifecycleLoggingHook = LoggingHook;
pub type RequestResponseToolLoggingHook = LoggingHook;

// ---------------------------------------------------------------------------
// 2. Retry hook with corrective feedback and per-run limit (agent_with_retry_hook)
// ---------------------------------------------------------------------------

static NEXT_RETRY_HOOK_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone, Default)]
struct RetryAttempts(HashMap<usize, usize>);

#[derive(Clone, Copy)]
enum RetryMode {
    Repeat,
    Feedback(&'static str),
}

/// Retries a completed, tool-free model turn from an [`AgentHook`]. Owns its
/// policy limit in the run-scoped [`HookContext`] scratchpad. Rig does not add
/// a separate retry counter to the agent: every retry consumes the request's
/// existing `max_turns` model-call budget.
pub struct RetryOnMarker {
    id: usize,
    marker: &'static str,
    max_retries: usize,
    mode: RetryMode,
}

impl RetryOnMarker {
    /// Build a retry hook that, on a marker, retries with corrective feedback.
    pub fn with_feedback(marker: &'static str, max_retries: usize, feedback: &'static str) -> Self {
        Self {
            id: NEXT_RETRY_HOOK_ID.fetch_add(1, Ordering::Relaxed),
            marker,
            max_retries,
            mode: RetryMode::Feedback(feedback),
        }
    }

    /// Build a retry hook that, on a marker, repeats the turn (discarding the
    /// rejected response and reusing the prompt and preceding history).
    pub fn repeat(marker: &'static str, max_retries: usize) -> Self {
        Self {
            id: NEXT_RETRY_HOOK_ID.fetch_add(1, Ordering::Relaxed),
            marker,
            max_retries,
            mode: RetryMode::Repeat,
        }
    }

    /// Convenience alias for [`RetryOnMarker::with_feedback`].
    pub fn new(marker: &'static str, max_retries: usize, feedback: &'static str) -> Self {
        Self::with_feedback(marker, max_retries, feedback)
    }

    /// Convenience alias mirroring the example constructor name.
    pub fn with_corrective_feedback(
        marker: &'static str,
        max_retries: usize,
        feedback: &'static str,
    ) -> Self {
        Self::with_feedback(marker, max_retries, feedback)
    }
}

// Clone is required for use as an AgentHook in stacks; the id is copied.
impl Clone for RetryOnMarker {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            marker: self.marker,
            max_retries: self.max_retries,
            mode: self.mode,
        }
    }
}

impl AgentHook for RetryOnMarker {
    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        let should_retry = event.content.iter().any(|content| {
            matches!(content, rig::message::AssistantContent::Text(text) if text.text.contains(self.marker))
        });
        if !should_retry {
            return ModelTurnAction::continue_run();
        }

        let attempt = ctx.scratchpad().update(|attempts: &mut RetryAttempts| {
            let attempt = attempts.0.entry(self.id).or_default();
            *attempt += 1;
            *attempt
        });
        if attempt > self.max_retries {
            return ModelTurnAction::stop(format!(
                "response retry limit ({}) exceeded",
                self.max_retries
            ));
        }

        println!(
            "[turn {}] rejected response; retry {attempt}/{}",
            event.turn, self.max_retries
        );
        match self.mode {
            RetryMode::Repeat => ModelTurnAction::repeat(),
            RetryMode::Feedback(feedback) => ModelTurnAction::retry_with_feedback(feedback),
        }
    }
}

// Aliases.
pub type RetryHook = RetryOnMarker;
pub type RetryOnMarkerHook = RetryOnMarker;

// ---------------------------------------------------------------------------
// 3. Tool failure outcome handling helpers (tool_result_outcomes)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    ReadDisk,
    ConnectNetwork,
}

impl Operation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadDisk => "read_disk",
            Self::ConnectNetwork => "connect_network",
        }
    }
}

#[derive(serde::Deserialize)]
pub struct ProbeArgs {
    pub operation: Operation,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, rig::ContextValue)]
#[context(key = "example.failure_site")]
pub struct FailureSite {
    pub operation: Operation,
    pub resource: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("disk read failed for /data/archive.bin")]
    DiskIo,
    #[error("network is unreachable for backup.example.net")]
    NetworkUnreachable,
    #[error("tool context: {0}")]
    Context(#[from] rig::tool::ToolContextError),
}

pub struct SystemProbe;

impl Tool for SystemProbe {
    const NAME: &'static str = "system_probe";
    type Error = ProbeError;
    type Args = ProbeArgs;
    type Output = String;

    fn description(&self) -> String {
        "Run a simulated system operation. Use read_disk for disk access or connect_network for remote access."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read_disk", "connect_network"]
                }
            },
            "required": ["operation"]
        })
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        match error {
            error @ ProbeError::DiskIo => ToolExecutionError::other(error.to_string())
                .with_model_feedback("the requested disk operation failed")
                .with_code("EIO")
                .with_retryable(false)
                .with_source(error),
            error @ ProbeError::NetworkUnreachable => {
                ToolExecutionError::network(error.to_string())
                    .with_model_feedback("the backup service is unreachable; try again later")
                    .with_code("ENETUNREACH")
                    .with_source(error)
            }
            ProbeError::Context(error) => ToolExecutionError::from(error),
        }
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let (error, site) = match args.operation {
            Operation::ReadDisk => (
                ProbeError::DiskIo,
                FailureSite {
                    operation: args.operation,
                    resource: "/data/archive.bin".to_string(),
                },
            ),
            Operation::ConnectNetwork => (
                ProbeError::NetworkUnreachable,
                FailureSite {
                    operation: args.operation,
                    resource: "backup.example.net".to_string(),
                },
            ),
        };
        context.insert_result(site)?;
        Err(error)
    }
}

/// Forces `system_probe` on the first turn only.
pub struct ForceSystemProbeOnFirstTurn;

pub fn system_probe_patch(turn: usize) -> Option<RequestPatch> {
    (turn == 1).then(|| {
        RequestPatch::new().tool_choice(ToolChoice::Specific {
            function_names: vec![tool_name::<SystemProbe>()],
        })
    })
}

impl AgentHook for ForceSystemProbeOnFirstTurn {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        match system_probe_patch(event.turn) {
            Some(patch) => CompletionCallAction::patch(patch),
            None => CompletionCallAction::continue_run(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FailureRecord {
    pub call_id: rig::message::CallId,
    pub tool_name: String,
    pub kind: ToolErrorKind,
    pub code: Option<String>,
    pub operation: Operation,
    pub resource: String,
}

#[derive(Clone, Default)]
pub struct FailureLedger(pub Vec<FailureRecord>);

/// Builds a [`FailureRecord`] from a failed tool result plus its typed context.
pub fn failure_record(
    call_id: &rig::message::CallId,
    tool_name: &str,
    result: &ToolResult,
    tool_context: &ToolContext,
) -> Option<FailureRecord> {
    let (Some(error), Ok(Some(site))) = (result.error(), tool_context.result::<FailureSite>())
    else {
        return None;
    };

    Some(FailureRecord {
        call_id: call_id.clone(),
        tool_name: tool_name.to_string(),
        kind: error.kind(),
        code: error.code().map(str::to_string),
        operation: site.operation,
        resource: site.resource,
    })
}

pub struct FailureRecorder;

impl AgentHook for FailureRecorder {
    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        // `on_outcome` fires for every effect family; only tool results matter here.
        let (Some(call_id), Some(tool_name), Some(result), Some(tool_context)) = (
            event.call_id,
            event.tool_name(),
            event.tool_result(),
            event.tool_context(),
        ) else {
            return OutcomeAction::proceed();
        };
        let Some(record) = failure_record(call_id, tool_name, result, tool_context) else {
            return OutcomeAction::proceed();
        };
        println!(
            "[recorder] {} {} failed: kind={}, code={}, resource={}",
            record.tool_name,
            record.operation.as_str(),
            record.kind,
            record.code.as_deref().unwrap_or("none"),
            record.resource
        );
        ctx.scratchpad()
            .update(|ledger: &mut FailureLedger| ledger.0.push(record));
        OutcomeAction::proceed()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum PolicyDecision {
    Fatal(String),
    Recoverable,
}

pub fn decide(record: &FailureRecord) -> PolicyDecision {
    if record.kind == ToolErrorKind::Other && record.code.as_deref() == Some("EIO") {
        PolicyDecision::Fatal(format!(
            "fatal disk I/O failure from {} ({})",
            record.tool_name, record.resource
        ))
    } else {
        PolicyDecision::Recoverable
    }
}

pub fn policy_action(ledger: Option<&FailureLedger>, call_id: &rig::message::CallId) -> OutcomeAction {
    let decision = ledger
        .and_then(|ledger| {
            ledger
                .0
                .iter()
                .rev()
                .find(|record| record.call_id == *call_id)
        })
        .map(decide);

    match decision {
        Some(PolicyDecision::Fatal(reason)) => OutcomeAction::stop(reason),
        Some(PolicyDecision::Recoverable) => {
            println!("[policy] recoverable failure; returning feedback to the model");
            OutcomeAction::proceed()
        }
        None => OutcomeAction::proceed(),
    }
}

pub struct FatalFailurePolicy;

impl AgentHook for FatalFailurePolicy {
    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let (Some(call_id), Some(result)) = (event.call_id, event.tool_result()) else {
            return OutcomeAction::proceed();
        };
        if result.error().is_none() {
            return OutcomeAction::proceed();
        }

        let ledger = ctx.scratchpad().get::<FailureLedger>();
        policy_action(ledger.as_ref(), call_id)
    }
}

// ---------------------------------------------------------------------------
// Tests (ported from tool_result_outcomes)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rig::ErrorKind;
    use rig::tool::ToolSet;

    fn stops_run(action: &OutcomeAction) -> bool {
        matches!(action, OutcomeAction::Replace(Err(report)) if report.kind == ErrorKind::Cancelled)
    }

    async fn structured_failure(operation: Operation) -> (ToolResult, ToolContext) {
        let tools = ToolSet::from_tools(vec![SystemProbe]);
        let mut context = ToolContext::new();
        let args = serde_json::json!({ "operation": operation }).to_string();
        let result = tools.execute(SystemProbe::NAME, args, &mut context).await;
        (result, context)
    }

    #[test]
    fn system_probe_is_forced_on_first_turn_only() {
        assert!(system_probe_patch(0).is_none());
        let first_turn = system_probe_patch(1);
        assert_eq!(
            first_turn.and_then(|patch| patch.tool_choice),
            Some(ToolChoice::Specific {
                function_names: vec![tool_name::<SystemProbe>()],
            })
        );
        assert!(system_probe_patch(2).is_none());
        assert!(system_probe_patch(3).is_none());
    }

    #[tokio::test]
    async fn connect_network_preserves_classification_and_typed_metadata() {
        let (result, context) = structured_failure(Operation::ConnectNetwork).await;
        assert!(result.error().is_some(), "probe should fail");
        let Some(error) = result.error() else {
            return;
        };
        assert_eq!(error.kind(), ToolErrorKind::Network);
        assert_eq!(error.code(), Some("ENETUNREACH"));
        assert_eq!(result.output().as_text(), error.model_feedback());
        let site = context.result::<FailureSite>();
        assert!(site.is_ok(), "decodes: {site:?}");
        assert_eq!(
            site.ok().flatten(),
            Some(FailureSite {
                operation: Operation::ConnectNetwork,
                resource: "backup.example.net".to_string(),
            })
        );
    }

    #[tokio::test]
    async fn recorder_data_drives_fatal_and_recoverable_actions_by_call_id() {
        let (fatal_result, fatal_context) = structured_failure(Operation::ReadDisk).await;
        let (recoverable_result, recoverable_context) =
            structured_failure(Operation::ConnectNetwork).await;

        let mut ledger = FailureLedger::default();
        let fatal_call = rig::message::CallId::from_wire("fatal");
        let recoverable_call = rig::message::CallId::from_wire("recoverable");
        let missing_call = rig::message::CallId::from_wire("missing");
        let fatal_record = failure_record(
            &fatal_call,
            SystemProbe::NAME,
            &fatal_result,
            &fatal_context,
        );
        let recoverable_record = failure_record(
            &recoverable_call,
            SystemProbe::NAME,
            &recoverable_result,
            &recoverable_context,
        );
        assert!(fatal_record.is_some(), "fatal failure record");
        assert!(recoverable_record.is_some(), "recoverable failure record");
        let (Some(fatal_record), Some(recoverable_record)) = (fatal_record, recoverable_record)
        else {
            return;
        };
        ledger.0.push(fatal_record);
        // Interleave a later recoverable record: policy must not use `last()`.
        ledger.0.push(recoverable_record);

        assert!(stops_run(&policy_action(Some(&ledger), &fatal_call)));
        assert!(matches!(
            policy_action(Some(&ledger), &recoverable_call),
            OutcomeAction::Proceed
        ));
        assert!(matches!(
            policy_action(Some(&ledger), &missing_call),
            OutcomeAction::Proceed
        ));
        assert!(matches!(
            policy_action(None, &fatal_call),
            OutcomeAction::Proceed
        ));
    }

    #[tokio::test]
    async fn missing_metadata_cannot_create_a_record_or_reuse_stale_state() {
        let (result, _context) = structured_failure(Operation::ReadDisk).await;
        assert!(
            failure_record(
                &rig::message::CallId::from_wire("other"),
                SystemProbe::NAME,
                &result,
                &ToolContext::new(),
            )
            .is_none()
        );

        let stale = FailureLedger(vec![FailureRecord {
            call_id: rig::message::CallId::from_wire("stale"),
            tool_name: SystemProbe::NAME.to_string(),
            kind: ToolErrorKind::Other,
            code: Some("EIO".to_string()),
            operation: Operation::ReadDisk,
            resource: "/stale".to_string(),
        }]);
        assert!(matches!(
            policy_action(Some(&stale), &rig::message::CallId::from_wire("fresh")),
            OutcomeAction::Proceed
        ));
    }
}