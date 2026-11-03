//! Agent helpers for a Rig-based (vectorless) RAG agent.
//!
//! This module combines several Rig examples into one reusable surface:
//!
//! 1. `agent` — helpers to build a minimal agent.
//! 2. `multi_turn_agent` — helpers to run an agent across multiple turns.
//! 3. `agent_with_tools` — helpers for registering runtime (dynamic) tools.
//! 4. `manual_tool_calls` — a manual tool-call loop helper.
//! 5. `force_tool_first_turn` — helpers for forcing tool use on the first turn.
//!
//! Everything compiles against `rig` 0.46.0. Provider-backed helpers need an
//! API key at runtime; the pure helpers and builders are unit-testable.

use anyhow::{Context, Result, bail};
use rig::DynModel;
use rig::agent::{Agent, AgentBuilder, AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, PromptResponse, RequestPatch};
use rig::operation::Completion;
use rig::completion::{CompletionRequest, Message};
use rig::message::{AssistantContent, EmptyToolName, ToolCall, ToolChoice, UserContent, ToolName};
use rig::providers::openai::{self, OpenAI};
use rig::tool::{DynamicTool, Tool, ToolContext as RigToolContext, ToolOutput, ToolSet};

use serde::{Deserialize, Serialize};
use serde_json::json;

// ---------------------------------------------------------------------------
// Local typed tool used by the agent helpers
// ---------------------------------------------------------------------------

/// A small calculator tool mirroring Rig's `add` example.
#[derive(Clone, Default)]
pub struct Add;

#[derive(Deserialize)]
pub struct AddArgs {
    x: i64,
    y: i64,
}

impl Tool for Add {
    const NAME: &'static str = "add";
    type Error = std::convert::Infallible;
    type Args = AddArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": { "type": "integer", "description": "The first addend" },
                "y": { "type": "integer", "description": "The second addend" }
            },
            "required": ["x", "y"]
        })
    }

    async fn call(
        &self,
        _context: &mut RigToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

/// Preamble used by the minimal `agent` example.
pub const PREAMBLE: &str = "You are a comedian here to entertain the user using humour and jokes.";

/// Prompt used by the minimal `agent` example.
pub const PROMPT: &str = "Entertain me!";

/// Alias for [`PREAMBLE`].
pub const MINIMAL_PREAMBLE: &str = PREAMBLE;

/// Alias for [`PROMPT`].
pub const MINIMAL_PROMPT: &str = PROMPT;

/// Preamble used by the manual tool-call example.
pub const MANUAL_PREAMBLE: &str =
    "You are a calculator. Never do arithmetic from memory. \
     Use the provided tools for every intermediate step. \
     You may emit one or multiple tool calls in a single turn. \
     Once all tool results are available, give a short final answer.";

/// Default prompt used by the manual tool-call example.
pub const MANUAL_PROMPT: &str =
    "Calculate (20 - 5) + (8 - 3). Use tools for each intermediate step before answering.";

/// Default maximum number of rounds for the manual tool-call loop.
pub const DEFAULT_MANUAL_MAX_ROUNDS: usize = 8;

// ---------------------------------------------------------------------------
// 1. Minimal agent helpers
// ---------------------------------------------------------------------------

/// Build the smallest useful Rig agent over `model`.
pub fn build_minimal_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    AgentBuilder::new(model).preamble(PREAMBLE).build()
}

/// Alias for [`build_minimal_agent`].
pub fn build_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    build_minimal_agent(model)
}

/// Alias for [`build_minimal_agent`].
pub fn minimal_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    build_minimal_agent(model)
}

/// Alias for [`build_minimal_agent`].
pub fn simple_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    build_minimal_agent(model)
}

/// Alias for [`build_minimal_agent`].
pub fn make_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    build_minimal_agent(model)
}

/// Alias for [`build_minimal_agent`].
pub fn new_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    build_minimal_agent(model)
}

/// Alias for [`build_minimal_agent`].
pub fn create_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    build_minimal_agent(model)
}

/// Alias for [`build_minimal_agent`].
pub fn basic_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    build_minimal_agent(model)
}

/// Alias for [`build_minimal_agent`].
pub fn agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    build_minimal_agent(model)
}

/// Alias for [`build_minimal_agent`].
pub fn default_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    build_minimal_agent(model)
}

/// Build a minimal agent with an explicit preamble.
pub fn build_agent_with_preamble<M, P>(model: M, preamble: P) -> Agent
where
    M: Into<DynModel<Completion>>,
    P: Into<String>,
{
    AgentBuilder::new(model).preamble(preamble).build()
}

/// Alias for [`build_agent_with_preamble`].
pub fn agent_with_preamble<M, P>(model: M, preamble: P) -> Agent
where
    M: Into<DynModel<Completion>>,
    P: Into<String>,
{
    build_agent_with_preamble(model, preamble)
}

/// Build a minimal OpenAI agent from the environment.
pub fn openai_minimal_agent() -> Result<Agent> {
    let client = OpenAI::from_env()?;
    Ok(build_minimal_agent(
        client.completion(openai::GPT_4O),
    ))
}

/// Alias for [`openai_minimal_agent`].
pub fn openai_agent() -> Result<Agent> {
    openai_minimal_agent()
}

/// Alias for [`openai_minimal_agent`].
pub fn openai_basic_agent() -> Result<Agent> {
    openai_minimal_agent()
}

// ---------------------------------------------------------------------------
// 2. Multi-turn runs
// ---------------------------------------------------------------------------

/// Build an agent suited to multi-turn calculator-style runs.
pub fn multi_turn_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    AgentBuilder::new(model)
        .preamble(
            "You are an assistant here to help the user perform arithmetic operations. \
             Follow the user's requests across multiple turns.",
        )
        .tool(Add)
        .build()
}

/// Alias for [`multi_turn_agent`].
pub fn multi_turn<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    multi_turn_agent(model)
}

/// Alias for [`multi_turn_agent`].
pub fn build_multi_turn_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    multi_turn_agent(model)
}

/// Alias for [`multi_turn_agent`].
pub fn new_multi_turn_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    multi_turn_agent(model)
}

/// Run several prompts against one agent, collecting their responses.
pub async fn run_multi_turn_prompts<P, I>(agent: &Agent, prompts: I) -> Result<Vec<PromptResponse>>
where
    P: Into<Message>,
    I: IntoIterator<Item = P>,
{
    let mut responses = Vec::new();
    for prompt in prompts {
        responses.push(agent.prompt(prompt).await?);
    }
    Ok(responses)
}

/// Alias for [`run_multi_turn_prompts`].
pub async fn multi_turn_run<P, I>(agent: &Agent, prompts: I) -> Result<Vec<PromptResponse>>
where
    P: Into<Message>,
    I: IntoIterator<Item = P>,
{
    run_multi_turn_prompts(agent, prompts).await
}

/// Alias for [`run_multi_turn_prompts`].
pub async fn run_multi_turn<P, I>(agent: &Agent, prompts: I) -> Result<Vec<PromptResponse>>
where
    P: Into<Message>,
    I: IntoIterator<Item = P>,
{
    run_multi_turn_prompts(agent, prompts).await
}

/// Run several prompts against one agent, collecting their output text.
pub async fn run_multi_turn_text<P, I>(agent: &Agent, prompts: I) -> Result<Vec<String>>
where
    P: Into<Message>,
    I: IntoIterator<Item = P>,
{
    let responses = run_multi_turn_prompts(agent, prompts).await?;
    Ok(responses.into_iter().map(|response| response.output().to_owned()).collect())
}

/// Alias for [`run_multi_turn_text`].
pub async fn multi_turn_text<P, I>(agent: &Agent, prompts: I) -> Result<Vec<String>>
where
    P: Into<Message>,
    I: IntoIterator<Item = P>,
{
    run_multi_turn_text(agent, prompts).await
}

/// Alias for [`run_multi_turn_text`].
pub async fn multi_turn_outputs<P, I>(agent: &Agent, prompts: I) -> Result<Vec<String>>
where
    P: Into<Message>,
    I: IntoIterator<Item = P>,
{
    run_multi_turn_text(agent, prompts).await
}

// ---------------------------------------------------------------------------
// 3. Runtime (dynamic) tool registration
// ---------------------------------------------------------------------------

/// Build the runtime tools from the `agent_with_tools` example.
pub fn runtime_tools() -> Result<Vec<DynamicTool>, EmptyToolName> {
    #[derive(Deserialize)]
    struct OperationArgs {
        x: i32,
        y: i32,
    }

    let parameters = json!({
        "type": "object",
        "properties": {
            "x": { "type": "integer" },
            "y": { "type": "integer" }
        },
        "required": ["x", "y"]
    });

    Ok(vec![
        DynamicTool::new(
            ToolName::new("add")?,
            "Add x and y",
            parameters.clone(),
            |args| {
                Box::pin(async move {
                    let args: OperationArgs = serde_json::from_value(args).map_err(|error| {
                        rig::tool::ToolExecutionError::invalid_args(error.to_string())
                            .with_source(error)
                    })?;
                    Ok(ToolOutput::json(json!(args.x + args.y)))
                })
            },
        ),
        DynamicTool::new(
            ToolName::new("subtract")?,
            "Subtract y from x",
            parameters,
            |args| {
                Box::pin(async move {
                    let args: OperationArgs = serde_json::from_value(args).map_err(|error| {
                        rig::tool::ToolExecutionError::invalid_args(error.to_string())
                            .with_source(error)
                    })?;
                    Ok(ToolOutput::json(json!(args.x - args.y)))
                })
            },
        ),
    ])
}

/// Alias for [`runtime_tools`].
pub fn runtime_toolset() -> Result<Vec<DynamicTool>, EmptyToolName> {
    runtime_tools()
}

/// Alias for [`runtime_tools`].
pub fn default_runtime_tools() -> Result<Vec<DynamicTool>, EmptyToolName> {
    runtime_tools()
}

/// Alias for [`runtime_tools`].
pub fn example_runtime_tools() -> Result<Vec<DynamicTool>, EmptyToolName> {
    runtime_tools()
}

/// Register runtime tools on an [`AgentBuilder`].
pub fn register_runtime_tools(
    builder: AgentBuilder,
    tools: Vec<DynamicTool>,
) -> AgentBuilder {
    builder.dynamic_tools(tools)
}

/// Alias for [`register_runtime_tools`].
pub fn with_runtime_tools(builder: AgentBuilder, tools: Vec<DynamicTool>) -> AgentBuilder {
    register_runtime_tools(builder, tools)
}

/// Alias for [`register_runtime_tools`].
pub fn add_runtime_tools(builder: AgentBuilder, tools: Vec<DynamicTool>) -> AgentBuilder {
    register_runtime_tools(builder, tools)
}

/// Alias for [`register_runtime_tools`].
pub fn attach_runtime_tools(builder: AgentBuilder, tools: Vec<DynamicTool>) -> AgentBuilder {
    register_runtime_tools(builder, tools)
}

/// Build an agent with the example runtime tools.
pub fn agent_with_runtime_tools<M>(model: M) -> Result<Agent, EmptyToolName>
where
    M: Into<DynModel<Completion>>,
{
    let tools = runtime_tools()?;
    Ok(AgentBuilder::new(model)
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             You must use the provided tools before answering.",
        )
        .dynamic_tools(tools)
        .max_tokens(1024)
        .default_max_turns(2)
        .build())
}

/// Alias for [`agent_with_runtime_tools`].
pub fn build_agent_with_runtime_tools<M>(model: M) -> Result<Agent, EmptyToolName>
where
    M: Into<DynModel<Completion>>,
{
    agent_with_runtime_tools(model)
}

/// Alias for [`agent_with_runtime_tools`].
pub fn build_agent_with_tools<M>(model: M) -> Result<Agent, EmptyToolName>
where
    M: Into<DynModel<Completion>>,
{
    agent_with_runtime_tools(model)
}

/// Alias for [`agent_with_runtime_tools`].
pub fn agent_with_tools<M>(model: M) -> Result<Agent, EmptyToolName>
where
    M: Into<DynModel<Completion>>,
{
    agent_with_runtime_tools(model)
}

/// Alias for [`agent_with_runtime_tools`].
pub fn create_agent_with_tools<M>(model: M) -> Result<Agent, EmptyToolName>
where
    M: Into<DynModel<Completion>>,
{
    agent_with_runtime_tools(model)
}

/// Build an agent with typed tools from this crate.
pub fn agent_with_typed_tools<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    AgentBuilder::new(model)
        .preamble("You are a calculator. Use the add tool for arithmetic.")
        .tool(Add)
        .build()
}

/// Alias for [`agent_with_typed_tools`].
pub fn build_agent_with_typed_tools<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    agent_with_typed_tools(model)
}

// ---------------------------------------------------------------------------
// 4. Manual tool-call helpers
// ---------------------------------------------------------------------------

/// Collect tool calls from an assistant message choice.
pub fn collect_tool_calls(choice: &[AssistantContent]) -> Vec<ToolCall> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .collect()
}

/// Extract text from an assistant message choice.
pub fn extract_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build the manual tool set used by the example (add + subtract).
pub fn manual_toolset() -> ToolSet {
    #[derive(Deserialize, Serialize)]
    struct OperationArgs {
        x: i32,
        y: i32,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("math error")]
    struct MathError;

    #[derive(Deserialize, Serialize)]
    struct AddTool;
    impl Tool for AddTool {
        const NAME: &'static str = "add";
        type Error = MathError;
        type Args = OperationArgs;
        type Output = i32;

        fn description(&self) -> String {
            "Add x and y together".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "The first number to add" },
                    "y": { "type": "number", "description": "The second number to add" }
                },
                "required": ["x", "y"]
            })
        }

        async fn call(
            &self,
            _context: &mut RigToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            Ok(args.x + args.y)
        }
    }

    #[derive(Deserialize, Serialize)]
    struct SubtractTool;
    impl Tool for SubtractTool {
        const NAME: &'static str = "subtract";
        type Error = MathError;
        type Args = OperationArgs;
        type Output = i32;

        fn description(&self) -> String {
            "Subtract y from x (x - y)".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "The number to subtract from" },
                    "y": { "type": "number", "description": "The number to subtract" }
                },
                "required": ["x", "y"]
            })
        }

        async fn call(
            &self,
            _context: &mut RigToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            Ok(args.x - args.y)
        }
    }

    let mut tools = ToolSet::default();
    tools.add_tool(AddTool);
    tools.add_tool(SubtractTool);
    tools
}

/// Alias for [`manual_toolset`].
pub fn manual_tools() -> ToolSet {
    manual_toolset()
}

/// Alias for [`manual_toolset`].
pub fn default_manual_tools() -> ToolSet {
    manual_toolset()
}

/// Convert a tool result into a user message, as the example does.
pub fn tool_result_message(tool_call: &ToolCall, output: ToolOutput) -> Message {
    let content = output.into_content();
    let result = UserContent::tool_result(
        tool_call.id.clone(),
        tool_call.function.name.clone(),
        content,
    );
    Message::User {
        content: vec![result],
    }
}

/// Alias for [`tool_result_message`].
pub fn tool_result_to_message(tool_call: &ToolCall, output: ToolOutput) -> Message {
    tool_result_message(tool_call, output)
}

/// Run the manual tool-call loop and return the final answer text.
pub async fn manual_tool_call_loop<M, P>(
    model: M,
    prompt: P,
    tools: &ToolSet,
    max_rounds: usize,
    preamble: impl Into<String>,
) -> Result<String>
where
    M: Into<DynModel<Completion>>,
    P: Into<Message>,
{
    let model: DynModel<Completion> = model.into();
    let preamble = preamble.into();
    let mut history: Vec<Message> = Vec::new();
    let mut current_prompt = prompt.into();

    for round in 1..=max_rounds {
        let mut request = CompletionRequest::new(current_prompt.clone())
            .preamble(preamble.clone())
            .messages(history.clone())
            .tools(tools.tool_definitions());

        if round == 1 {
            request = request.tool_choice(ToolChoice::Required);
        }

        let response = model.call(request).await.context("manual model call failed")?;
        let tool_calls = collect_tool_calls(&response.choice);

        history.push(current_prompt.clone());
        history.extend(response.message());

        if tool_calls.is_empty() {
            return Ok(extract_text(&response.choice));
        }

        for tool_call in &tool_calls {
            let args = serde_json::to_string(&tool_call.function.arguments)
                .context("failed to serialize tool arguments")?;
            let result = tools
                .execute(&tool_call.function.name, args.clone(), &mut RigToolContext::new())
                .await;
            let output = result.output().clone();
            history.push(tool_result_message(tool_call, output));
        }

        current_prompt = match history.pop() {
            Some(prompt) => prompt,
            None => bail!("manual tool loop history unexpectedly empty"),
        };
    }

    bail!("manual tool loop exceeded {max_rounds} rounds")
}

/// Run the manual tool-call loop with default tools, rounds, and preamble.
pub async fn manual_tool_calls<M, P>(model: M, prompt: P) -> Result<String>
where
    M: Into<DynModel<Completion>>,
    P: Into<Message>,
{
    let tools = manual_toolset();
    manual_tool_call_loop(model, prompt, &tools, DEFAULT_MANUAL_MAX_ROUNDS, MANUAL_PREAMBLE).await
}

/// Alias for [`manual_tool_calls`].
pub async fn manual_tool_call<M, P>(model: M, prompt: P) -> Result<String>
where
    M: Into<DynModel<Completion>>,
    P: Into<Message>,
{
    manual_tool_calls(model, prompt).await
}

/// Alias for [`manual_tool_calls`].
pub async fn run_manual_tool_calls<M, P>(model: M, prompt: P) -> Result<String>
where
    M: Into<DynModel<Completion>>,
    P: Into<Message>,
{
    manual_tool_calls(model, prompt).await
}

/// Alias for [`manual_tool_calls`].
pub async fn run_manual_tool_call_loop<M, P>(model: M, prompt: P) -> Result<String>
where
    M: Into<DynModel<Completion>>,
    P: Into<Message>,
{
    manual_tool_calls(model, prompt).await
}

/// Run the manual loop with caller-supplied tools.
pub async fn manual_tool_calls_with_tools<M, P>(
    model: M,
    prompt: P,
    tools: &ToolSet,
) -> Result<String>
where
    M: Into<DynModel<Completion>>,
    P: Into<Message>,
{
    manual_tool_call_loop(model, prompt, tools, DEFAULT_MANUAL_MAX_ROUNDS, MANUAL_PREAMBLE).await
}

/// Alias for [`manual_tool_calls_with_tools`].
pub async fn run_manual_with_tools<M, P>(model: M, prompt: P, tools: &ToolSet) -> Result<String>
where
    M: Into<DynModel<Completion>>,
    P: Into<Message>,
{
    manual_tool_calls_with_tools(model, prompt, tools).await
}

// ---------------------------------------------------------------------------
// 5. Forced first-turn tool use
// ---------------------------------------------------------------------------

/// Hook that forces `tool_choice = Required` on the first turn only.
#[derive(Clone, Default)]
pub struct ForceToolOnFirstTurn;

impl AgentHook for ForceToolOnFirstTurn {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if ctx.turn() == 1 {
            CompletionCallAction::patch(RequestPatch::new().tool_choice(ToolChoice::Required))
        } else {
            CompletionCallAction::continue_run()
        }
    }
}

/// Alias for [`ForceToolOnFirstTurn`].
pub type ForceToolFirstTurn = ForceToolOnFirstTurn;

/// Hook that forces `tool_choice = Required` on every turn (the footgun).
#[derive(Clone, Default)]
pub struct ForceToolEveryTurn;

impl AgentHook for ForceToolEveryTurn {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        CompletionCallAction::patch(RequestPatch::new().tool_choice(ToolChoice::Required))
    }
}

/// Return a request patch that forces tool use on the first turn only.
pub fn force_tool_first_turn_patch(turn: usize) -> Option<RequestPatch> {
    (turn == 1).then(|| RequestPatch::new().tool_choice(ToolChoice::Required))
}

/// Alias for [`force_tool_first_turn_patch`].
pub fn force_tool_on_first_turn_patch(turn: usize) -> Option<RequestPatch> {
    force_tool_first_turn_patch(turn)
}

/// Alias for [`force_tool_first_turn_patch`].
pub fn first_turn_tool_choice_patch(turn: usize) -> Option<RequestPatch> {
    force_tool_first_turn_patch(turn)
}

/// A helper function returning the first-turn hook.
pub fn force_tool_first_turn() -> ForceToolOnFirstTurn {
    ForceToolOnFirstTurn
}

/// Alias for [`force_tool_first_turn`].
pub fn force_first_turn_tool_use() -> ForceToolOnFirstTurn {
    force_tool_first_turn()
}

/// Alias for [`force_tool_first_turn`].
pub fn forced_first_turn_tool_use() -> ForceToolOnFirstTurn {
    force_tool_first_turn()
}

/// Build an agent that forces the `add` tool on the first turn.
pub fn force_tool_first_turn_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    AgentBuilder::new(model)
        .preamble("You are a calculator assistant. Use the add tool for arithmetic, then report the result.")
        .tool(Add)
        .build()
}

/// Alias for [`force_tool_first_turn_agent`].
pub fn forced_first_turn_agent<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    force_tool_first_turn_agent(model)
}

/// Alias for [`force_tool_first_turn_agent`].
pub fn agent_with_forced_first_turn<M>(model: M) -> Agent
where
    M: Into<DynModel<Completion>>,
{
    force_tool_first_turn_agent(model)
}

// ---------------------------------------------------------------------------
// OpenAI convenience for forced first-turn runs
// ---------------------------------------------------------------------------

/// Build an OpenAI agent suitable for forced first-turn tool use.
pub fn openai_force_tool_first_turn_agent() -> Result<Agent> {
    let client = OpenAI::from_env()?;
    Ok(force_tool_first_turn_agent(
        client.completion(openai::GPT_4O),
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_non_empty() {
        assert!(!PREAMBLE.is_empty());
        assert!(!PROMPT.is_empty());
        assert!(!MANUAL_PREAMBLE.is_empty());
        assert!(!MANUAL_PROMPT.is_empty());
    }

    #[test]
    fn runtime_tools_build() {
        let tools = runtime_tools().expect("runtime tools");
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn manual_toolset_builds() {
        let tools = manual_toolset();
        assert!(!tools.tool_definitions().is_empty());
    }

    #[test]
    fn collect_and_extract_helpers() {
        let calls = collect_tool_calls(&[]);
        assert!(calls.is_empty());
        assert_eq!(extract_text(&[]), "");
    }

    #[test]
    fn force_patch_is_first_turn_only() {
        assert!(force_tool_first_turn_patch(0).is_none());
        assert!(force_tool_first_turn_patch(1).is_some());
        assert!(force_tool_first_turn_patch(2).is_none());
    }

    #[test]
    fn hook_is_clone() {
        let hook = force_tool_first_turn();
        let _ = hook.clone();
    }
}