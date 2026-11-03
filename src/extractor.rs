//! Typed extraction helpers for a Rig-based vectorless RAG agent.
//!
//! Mirrors the Rig `extractor` example: define a serializable target struct,
//! build a Rig [`Extractor`] over it, and run it against a piece of text to
//! get a typed value back. No embeddings or vector store are involved; the
//! text is handed to the model directly.
//!
//! Requires a completion model (and, at runtime, a provider API key).

use anyhow::Result;
use rig::DynModel;
use rig::operation::Completion;
use rig::extractor::{Extractor, ExtractorBuilder};
use rig::message::Message;
use rig::schemars::JsonSchema;
use rig::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;

/// A small serializable struct used as the extraction target example,
/// mirroring the Rig `extractor` example's `Person`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
pub struct Person {
    /// The person's first name, if stated.
    #[schemars(required)]
    pub first_name: Option<String>,
    /// The person's last name, if stated.
    #[schemars(required)]
    pub last_name: Option<String>,
    /// The person's job or occupation, if stated.
    #[schemars(required)]
    pub job: Option<String>,
}

impl Person {
    /// Create a `Person` from optional field values.
    pub fn new(
        first_name: Option<String>,
        last_name: Option<String>,
        job: Option<String>,
    ) -> Self {
        Self {
            first_name,
            last_name,
            job,
        }
    }
}

/// Alias for [`Person`].
pub type ExtractedPerson = Person;
/// Alias for [`Person`].
pub type PersonData = Person;
/// Alias for [`Person`].
pub type Contact = Person;
/// Alias for [`Person`].
pub type Entity = Person;

// ---------------------------------------------------------------------------
// Extractor construction
// ---------------------------------------------------------------------------

/// Build a Rig [`Extractor`] for [`Person`] over `model`.
pub fn build_extractor<M>(model: M) -> Extractor<Person>
where
    M: Into<DynModel<Completion>>,
{
    ExtractorBuilder::<Person>::new(model).build()
}

/// Alias for [`build_extractor`].
pub fn extractor<M>(model: M) -> Extractor<Person>
where
    M: Into<DynModel<Completion>>,
{
    build_extractor(model)
}

/// Alias for [`build_extractor`].
pub fn new_extractor<M>(model: M) -> Extractor<Person>
where
    M: Into<DynModel<Completion>>,
{
    build_extractor(model)
}

/// Alias for [`build_extractor`].
pub fn create_extractor<M>(model: M) -> Extractor<Person>
where
    M: Into<DynModel<Completion>>,
{
    build_extractor(model)
}

/// Alias for [`build_extractor`].
pub fn make_extractor<M>(model: M) -> Extractor<Person>
where
    M: Into<DynModel<Completion>>,
{
    build_extractor(model)
}

/// Alias for [`build_extractor`].
pub fn person_extractor<M>(model: M) -> Extractor<Person>
where
    M: Into<DynModel<Completion>>,
{
    build_extractor(model)
}

/// Build a Rig [`Extractor`] for an arbitrary serializable target type `T`.
pub fn build_extractor_for<T, M>(model: M) -> Extractor<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + WasmCompatSend + WasmCompatSync + 'static,
    M: Into<DynModel<Completion>>,
{
    ExtractorBuilder::<T>::new(model).build()
}

/// Alias for [`build_extractor_for`].
pub fn extractor_for<T, M>(model: M) -> Extractor<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + WasmCompatSend + WasmCompatSync + 'static,
    M: Into<DynModel<Completion>>,
{
    build_extractor_for(model)
}

/// Alias for [`build_extractor_for`].
pub fn new_extractor_for<T, M>(model: M) -> Extractor<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + WasmCompatSend + WasmCompatSync + 'static,
    M: Into<DynModel<Completion>>,
{
    build_extractor_for(model)
}

/// Alias for [`build_extractor_for`].
pub fn create_extractor_for<T, M>(model: M) -> Extractor<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + WasmCompatSend + WasmCompatSync + 'static,
    M: Into<DynModel<Completion>>,
{
    build_extractor_for(model)
}

// ---------------------------------------------------------------------------
// Running an extractor
// ---------------------------------------------------------------------------

/// Run a Rig extractor for [`Person`] over `text`, returning the typed value.
pub async fn extract_from_text<M, I>(model: M, text: I) -> Result<Person>
where
    M: Into<DynModel<Completion>>,
    I: Into<Message>,
{
    run_extractor::<Person, M, I>(model, text).await
}

/// Run a Rig extractor for an arbitrary type `T` over `text`.
pub async fn run_extractor<T, M, I>(model: M, text: I) -> Result<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + WasmCompatSend + WasmCompatSync + 'static,
    M: Into<DynModel<Completion>>,
    I: Into<Message>,
{
    let extractor = build_extractor_for::<T, M>(model);
    let response = extractor.extract(text).await?;
    Ok(response.output)
}

/// Alias for [`extract_from_text`].
pub async fn extract_person<M, I>(model: M, text: I) -> Result<Person>
where
    M: Into<DynModel<Completion>>,
    I: Into<Message>,
{
    extract_from_text(model, text).await
}

/// Alias for [`extract_from_text`].
pub async fn extract_person_from_text<M, I>(model: M, text: I) -> Result<Person>
where
    M: Into<DynModel<Completion>>,
    I: Into<Message>,
{
    extract_from_text(model, text).await
}

/// Alias for [`extract_from_text`].
pub async fn extract_text<M, I>(model: M, text: I) -> Result<Person>
where
    M: Into<DynModel<Completion>>,
    I: Into<Message>,
{
    extract_from_text(model, text).await
}

/// Alias for [`extract_from_text`].
pub async fn extract<M, I>(model: M, text: I) -> Result<Person>
where
    M: Into<DynModel<Completion>>,
    I: Into<Message>,
{
    extract_from_text(model, text).await
}

/// Alias for [`extract_from_text`].
pub async fn extract_structured<M, I>(model: M, text: I) -> Result<Person>
where
    M: Into<DynModel<Completion>>,
    I: Into<Message>,
{
    extract_from_text(model, text).await
}

/// Alias for [`extract_from_text`].
pub async fn extract_typed<M, I>(model: M, text: I) -> Result<Person>
where
    M: Into<DynModel<Completion>>,
    I: Into<Message>,
{
    extract_from_text(model, text).await
}

/// Alias for [`run_extractor`].
pub async fn extract_typed_value<T, M, I>(model: M, text: I) -> Result<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + WasmCompatSend + WasmCompatSync + 'static,
    M: Into<DynModel<Completion>>,
    I: Into<Message>,
{
    run_extractor(model, text).await
}

/// Alias for [`run_extractor`].
pub async fn extract_value<T, M, I>(model: M, text: I) -> Result<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + WasmCompatSend + WasmCompatSync + 'static,
    M: Into<DynModel<Completion>>,
    I: Into<Message>,
{
    run_extractor(model, text).await
}

// ---------------------------------------------------------------------------
// OpenAI helpers (mirror the Rig example)
// ---------------------------------------------------------------------------

/// Build a [`Person`] extractor using an OpenAI client from the environment.
pub fn openai_person_extractor() -> Result<Extractor<Person>> {
    let client = rig::providers::openai::OpenAI::from_env()?;
    Ok(ExtractorBuilder::<Person>::new(client.completion(rig::providers::openai::GPT_4)).build())
}

/// Alias for [`openai_person_extractor`].
pub fn openai_extractor() -> Result<Extractor<Person>> {
    openai_person_extractor()
}

/// Alias for [`openai_person_extractor`].
pub fn extractor_from_openai_env() -> Result<Extractor<Person>> {
    openai_person_extractor()
}

/// Run extraction for [`Person`] using an OpenAI client from the environment.
pub async fn extract_from_text_openai(text: &str) -> Result<Person> {
    let extractor = openai_person_extractor()?;
    let response = extractor.extract(text).await?;
    Ok(response.output)
}

/// Alias for [`extract_from_text_openai`].
pub async fn extract_person_openai(text: &str) -> Result<Person> {
    extract_from_text_openai(text).await
}

// Re-export Rig extractor types for convenience.
pub use rig::extractor::Extractor as RigExtractor;
pub use rig::extractor::ExtractorBuilder as RigExtractorBuilder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn person_roundtrip() {
        let p = Person {
            first_name: Some("John".into()),
            last_name: Some("Doe".into()),
            job: Some("software engineer".into()),
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: Person = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn person_default() {
        let p = Person::default();
        assert!(p.first_name.is_none());
        assert!(p.last_name.is_none());
        assert!(p.job.is_none());
    }

    #[test]
    fn person_new() {
        let p = Person::new(Some("Jane".into()), Some("Smith".into()), Some("data scientist".into()));
        assert_eq!(p.first_name.as_deref(), Some("Jane"));
        assert_eq!(p.last_name.as_deref(), Some("Smith"));
        assert_eq!(p.job.as_deref(), Some("data scientist"));
    }

    #[test]
    fn aliases_are_person() {
        let p = Person::default();
        let _: ExtractedPerson = p.clone();
        let _: PersonData = p.clone();
        let _: Contact = p.clone();
        let _: Entity = p;
    }
}
