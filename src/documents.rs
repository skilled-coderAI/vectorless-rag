//! Vectorless document handling utilities for a Rig-based RAG agent.
//!
//! Provides a small, dependency-light [`Document`] type plus helpers to load
//! plain text (and, when the `pdf` feature is enabled, PDF) files and to
//! render a slice of documents into a compact context string suitable for an
//! agent preamble or `.context(...)` call. No embeddings or vector store are
//! involved: retrieval is expected to happen by direct inspection of these
//! documents.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// A loaded document. Carries its own id and title alongside the extracted
/// text and the path it was loaded from.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Document {
    /// Stable identifier for the document (defaults to its source path).
    pub id: String,
    /// Human-readable title (defaults to the file stem).
    pub title: String,
    /// Extracted text content.
    pub text: String,
    /// Path the document was loaded from.
    pub source_path: PathBuf,
}

impl Document {
    /// Create a document, deriving a title from `source_path` when `title` is
    /// empty.
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        text: impl Into<String>,
        source_path: impl Into<PathBuf>,
    ) -> Self {
        let title = title.into();
        let source_path: PathBuf = source_path.into();
        let title = if title.is_empty() {
            source_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("untitled")
                .to_owned()
        } else {
            title
        };
        Self {
            id: id.into(),
            title,
            text: text.into(),
            source_path,
        }
    }
}

/// Load a UTF-8 text file into a [`Document`].
pub fn load_text_file<P: AsRef<Path>>(path: P) -> Result<Document> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read text file {}", path.display()))?;
    let title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled")
        .to_owned();
    Ok(Document {
        id: path.display().to_string(),
        title,
        text,
        source_path: path.to_path_buf(),
    })
}

/// Load a PDF file into a single [`Document`] whose text is the concatenation
/// of all of its pages.
pub fn load_pdf<P: AsRef<Path>>(path: P) -> Result<Document> {
    let path = path.as_ref();
    let chunks = pdf_text_chunks(path)?;
    if chunks.is_empty() {
        anyhow::bail!("no content found in PDF file: {}", path.display());
    }
    let title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled")
        .to_owned();
    Ok(Document {
        id: path.display().to_string(),
        title,
        text: chunks.join("\n\n"),
        source_path: path.to_path_buf(),
    })
}

/// Load a PDF file into one [`Document`] per extracted text chunk.
pub fn load_pdf_chunks<P: AsRef<Path>>(path: P) -> Result<Vec<Document>> {
    let path = path.as_ref();
    let chunks = pdf_text_chunks(path)?;
    if chunks.is_empty() {
        anyhow::bail!("no content found in PDF file: {}", path.display());
    }
    let base_title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled")
        .to_owned();
    Ok(chunks
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| Document {
            id: format!("{}#{}", path.display(), i),
            title: format!("{} (chunk {})", base_title, i),
            text: chunk,
            source_path: path.to_path_buf(),
        })
        .collect())
}

fn pdf_text_chunks(path: &Path) -> Result<Vec<String>> {
    const CHUNK_SIZE: usize = 2000;
    let raw = rig::loaders::PdfFileLoader::with_glob(path.to_str().context("invalid path")?)?
        .read()
        .into_iter()
        .filter_map(|result| {
            result
                .map_err(|e| {
                    eprintln!("Error reading PDF content: {e}");
                    e
                })
                .ok()
        })
        .flat_map(|content| {
            let mut chunks = Vec::new();
            let mut current = String::new();
            for word in content.split_whitespace() {
                if current.len() + word.len() + 1 > CHUNK_SIZE && !current.is_empty() {
                    chunks.push(std::mem::take(&mut current).trim().to_string());
                }
                current.push_str(word);
                current.push(' ');
            }
            if !current.is_empty() {
                chunks.push(current.trim().to_string());
            }
            chunks
        })
        .collect::<Vec<_>>();
    Ok(raw)
}

/// Build a compact context string from a slice of documents, suitable for an
/// agent preamble or for passing to `AgentBuilder::context`.
pub fn build_context(docs: &[Document]) -> String {
    build_context_with_limit(docs, 8_000)
}

/// Build a context string from `docs`, truncating the total output to
/// `max_chars` characters.
pub fn build_context_with_limit(docs: &[Document], max_chars: usize) -> String {
    let mut out = String::new();
    for (i, doc) in docs.iter().enumerate() {
        let header = format!("### {}\n(id: {})\n", doc.title, doc.id);
        let mut text = doc.text.clone();
        // Keep each document contribution modest so the preamble stays small.
        if text.chars().count() > 1_000 {
            text = text.chars().take(1_000).collect::<String>() + "…";
        }
        let block = format!("{header}{}\n", text);
        if out.chars().count() + block.chars().count() > max_chars {
            // Truncate the final block to fit the budget, if possible.
            let remaining = max_chars.saturating_sub(out.chars().count());
            if remaining > header.chars().count() + 4 {
                let take = remaining.saturating_sub(header.chars().count() + 2);
                let truncated: String = text.chars().take(take).collect();
                out.push_str(&format!("{header}{}…\n", truncated));
            }
            break;
        }
        out.push_str(&block);
        if i + 1 < docs.len() {
            out.push('\n');
        }
    }
    out
}

/// Alias for [`build_context`].
pub fn context_from_documents(docs: &[Document]) -> String {
    build_context(docs)
}

/// Alias for [`build_context`].
pub fn documents_to_context(docs: &[Document]) -> String {
    build_context(docs)
}

/// Alias for [`build_context`].
pub fn context_string(docs: &[Document]) -> String {
    build_context(docs)
}

/// Combine a base preamble with the rendered context of `docs`.
pub fn preamble_with_documents(base: &str, docs: &[Document]) -> String {
    let context = build_context(docs);
    if context.is_empty() {
        base.to_owned()
    } else {
        format!("{base}\n\nContext:\n{context}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_text_file() {
        let dir = std::env::temp_dir().join("vectorless-rag-docs-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hello.txt");
        std::fs::write(&path, "hello world").unwrap();
        let doc = load_text_file(&path).unwrap();
        assert_eq!(doc.title, "hello");
        assert_eq!(doc.text, "hello world");
        assert_eq!(doc.source_path, path);
        let ctx = build_context(&[doc]);
        assert!(ctx.contains("hello world"));
    }
}
