//! Vectorless retrieval utilities for a Rig-based RAG agent.
//!
//! Unlike embedding-based retrieval, this module scores documents with simple
//! lexical/keyword matching: no embeddings, no vector store. A [`Retriever`]
//! holds a collection of [`crate::documents::Document`]s and can return the
//! ones most relevant to a query, plus helpers to fold them into a context
//! string suitable for an agent preamble or `.context(...)` call.
//!
//! The shape mirrors Rig's `chain` example (retrieve context, fold it into the
//! prompt, then prompt the agent) but replaces the vector-store lookup with a
//! plain keyword search.

use crate::documents::Document;

/// A vectorless retriever: a collection of documents searched by keyword.
#[derive(Debug, Clone, Default)]
pub struct Retriever {
    /// The documents available for retrieval.
    pub documents: Vec<Document>,
}

impl Retriever {
    /// Create a retriever from a list of documents.
    pub fn new(documents: Vec<Document>) -> Self {
        Self { documents }
    }

    /// Alias for [`Retriever::new`].
    pub fn from_documents(documents: Vec<Document>) -> Self {
        Self { documents }
    }

    /// Alias for [`Retriever::new`].
    pub fn from_docs(docs: Vec<Document>) -> Self {
        Self { documents: docs }
    }

    /// Alias for [`Retriever::new`].
    pub fn with_documents(documents: Vec<Document>) -> Self {
        Self { documents }
    }

    /// Alias for [`Retriever::new`].
    pub fn with_docs(docs: Vec<Document>) -> Self {
        Self { documents: docs }
    }

    /// The underlying documents.
    pub fn docs(&self) -> &[Document] {
        &self.documents
    }

    /// Mutable access to the underlying documents.
    pub fn docs_mut(&mut self) -> &mut Vec<Document> {
        &mut self.documents
    }

    /// Number of documents held.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Whether the retriever holds no documents.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Add a document.
    pub fn add(&mut self, document: Document) {
        self.documents.push(document);
    }

    /// Alias for [`Retriever::add`].
    pub fn push(&mut self, document: Document) {
        self.documents.push(document);
    }

    /// Simple keyword/lexical search over the held documents.
    pub fn search(&self, query: &str) -> Vec<Document> {
        search_documents(&self.documents, query)
    }

    /// Alias for [`Retriever::search`].
    pub fn retrieve(&self, query: &str) -> Vec<Document> {
        self.search(query)
    }

    /// Alias for [`Retriever::search`].
    pub fn lookup(&self, query: &str) -> Vec<Document> {
        self.search(query)
    }

    /// Alias for [`Retriever::search`].
    pub fn find(&self, query: &str) -> Vec<Document> {
        self.search(query)
    }

    /// Search and keep at most `limit` results.
    pub fn search_top(&self, query: &str, limit: usize) -> Vec<Document> {
        let mut results = self.search(query);
        results.truncate(limit);
        results
    }

    /// Alias for [`Retriever::search_top`].
    pub fn top(&self, query: &str, limit: usize) -> Vec<Document> {
        self.search_top(query, limit)
    }

    /// Format a slice of retrieved documents into a context string.
    pub fn format_context(&self, docs: &[Document]) -> String {
        format_context(docs)
    }

    /// Alias for [`Retriever::format_context`].
    pub fn build_context(&self, docs: &[Document]) -> String {
        build_context(docs)
    }

    /// Alias for [`Retriever::format_context`].
    pub fn context(&self, docs: &[Document]) -> String {
        format_context(docs)
    }

    /// Retrieve documents for `query` and format them into a context string.
    pub fn search_context(&self, query: &str) -> String {
        format_context(&self.search(query))
    }

    /// Alias for [`Retriever::search_context`].
    pub fn retrieve_context(&self, query: &str) -> String {
        self.search_context(query)
    }

    /// Alias for [`Retriever::search_context`].
    pub fn lookup_context(&self, query: &str, prompt: &str) -> String {
        lookup_context(self.search(query), prompt)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Tokenize text into lowercase alphanumeric terms.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Score a single document against a query using lexical term overlap.
pub fn score_document(doc: &Document, query: &str) -> usize {
    let q_tokens = tokenize(query);
    if q_tokens.is_empty() {
        return 0;
    }
    let haystack = format!("{} {}", doc.title, doc.text);
    let haystack_lower = haystack.to_lowercase();
    let doc_tokens = tokenize(&haystack);

    let mut score = 0usize;
    for qt in &q_tokens {
        let occurrences = doc_tokens
            .iter()
            .filter(|dt| dt.as_str() == qt.as_str())
            .count();
        score += occurrences;
        // Reward substring matches that did not tokenize identically.
        if occurrences == 0 && haystack_lower.contains(qt) {
            score += 1;
        }
    }

    // Phrase bonus.
    let q_lower = query.to_lowercase();
    let trimmed = q_lower.trim();
    if !trimmed.is_empty() && haystack_lower.contains(trimmed) {
        score += q_tokens.len().saturating_mul(2);
    }

    score
}

/// Simple keyword/lexical search over a slice of documents.
pub fn search_documents(docs: &[Document], query: &str) -> Vec<Document> {
    if query.trim().is_empty() {
        return docs.to_vec();
    }

    let mut scored: Vec<(usize, usize, Document)> = docs
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, doc)| (score_document(&doc, query), i, doc))
        .collect();

    scored.retain(|(score, _, _)| *score > 0);
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

    scored.into_iter().map(|(_, _, doc)| doc).collect()
}

/// Alias for [`search_documents`].
pub fn search(docs: &[Document], query: &str) -> Vec<Document> {
    search_documents(docs, query)
}

/// Alias for [`search_documents`].
pub fn lexical_search(docs: &[Document], query: &str) -> Vec<Document> {
    search_documents(docs, query)
}

/// Alias for [`search_documents`].
pub fn keyword_search(docs: &[Document], query: &str) -> Vec<Document> {
    search_documents(docs, query)
}

/// Alias for [`search_documents`].
pub fn retrieve(docs: &[Document], query: &str) -> Vec<Document> {
    search_documents(docs, query)
}

/// Alias for [`search_documents`].
pub fn lookup(docs: &[Document], query: &str) -> Vec<Document> {
    search_documents(docs, query)
}

/// Search and keep at most `limit` results.
pub fn search_top(docs: &[Document], query: &str, limit: usize) -> Vec<Document> {
    let mut results = search_documents(docs, query);
    results.truncate(limit);
    results
}

/// Alias for [`search_top`].
pub fn top(docs: &[Document], query: &str, limit: usize) -> Vec<Document> {
    search_top(docs, query, limit)
}

/// Format retrieved documents into a context string.
pub fn format_context(docs: &[Document]) -> String {
    build_context(docs)
}

/// Alias for [`format_context`].
pub fn build_context(docs: &[Document]) -> String {
    crate::documents::build_context(docs)
}

/// Alias for [`format_context`] with a character limit.
pub fn format_context_with_limit(docs: &[Document], max_chars: usize) -> String {
    build_context_with_limit(docs, max_chars)
}

/// Alias for [`format_context_with_limit`].
pub fn build_context_with_limit(docs: &[Document], max_chars: usize) -> String {
    crate::documents::build_context_with_limit(docs, max_chars)
}

/// Alias for [`format_context`].
pub fn context_from_documents(docs: &[Document]) -> String {
    format_context(docs)
}

/// Alias for [`format_context`].
pub fn documents_to_context(docs: &[Document]) -> String {
    format_context(docs)
}

/// Alias for [`format_context`].
pub fn context_string(docs: &[Document]) -> String {
    format_context(docs)
}

/// Alias for [`format_context`].
pub fn format_docs(docs: &[Document]) -> String {
    format_context(docs)
}

/// Alias for [`format_context`].
pub fn format_retrieved(docs: &[Document]) -> String {
    format_context(docs)
}

/// Alias for [`format_context`].
pub fn render_context(docs: &[Document]) -> String {
    format_context(docs)
}

/// Mirror of Rig's `chain` example helper: fold retrieved docs into a prompt.
pub fn lookup_context(docs: Vec<Document>, prompt: &str) -> String {
    crate::documents::build_context(&docs) + "\n\n" + prompt
}

/// Alias for [`lookup_context`] taking a slice.
pub fn lookup_context_slice(docs: &[Document], prompt: &str) -> String {
    format!("{}\n\n{}", crate::documents::build_context(docs), prompt)
}

/// Combine a base preamble with the rendered context of retrieved docs.
pub fn preamble_with_documents(base: &str, docs: &[Document]) -> String {
    crate::documents::preamble_with_documents(base, docs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn sample_docs() -> Vec<Document> {
        vec![
            Document {
                id: "1".into(),
                title: "flurbo".into(),
                text: "A flurbo is a green alien that lives on cold planets".into(),
                source_path: PathBuf::from("flurbo.txt"),
            },
            Document {
                id: "2".into(),
                title: "glarb-glarb".into(),
                text: "A glarb-glarb is an ancient tool used by farmers".into(),
                source_path: PathBuf::from("glarb.txt"),
            },
            Document {
                id: "3".into(),
                title: "linglingdong".into(),
                text: "A term used by inhabitants of the far side of the moon".into(),
                source_path: PathBuf::from("ling.txt"),
            },
        ]
    }

    #[test]
    fn search_finds_relevant() {
        let retriever = Retriever::new(sample_docs());
        let results = retriever.search("glarb-glarb");
        assert_eq!(results.len(), 1);
        assert!(results[0].title.contains("glarb"));
    }

    #[test]
    fn search_empty_query_returns_all() {
        let retriever = Retriever::new(sample_docs());
        assert_eq!(retriever.search("").len(), 3);
    }

    #[test]
    fn format_context_contains_text() {
        let retriever = Retriever::new(sample_docs());
        let results = retriever.search("flurbo");
        let ctx = retriever.format_context(&results);
        assert!(ctx.contains("green alien"));
    }

    #[test]
    fn free_functions_work() {
        let docs = sample_docs();
        let results = search(&docs, "ancient tool");
        assert_eq!(results.len(), 1);
        let ctx = format_context(&results);
        assert!(ctx.contains("glarb"));
    }

    #[test]
    fn lookup_context_folds_prompt() {
        let docs = sample_docs();
        let out = lookup_context(docs.clone(), "What is a glarb-glarb?");
        // search separately
        let _ = docs;
        assert!(out.contains("What is a glarb-glarb?"));
    }
}
