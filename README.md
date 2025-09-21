# Quipu

A high-performance semantic indexing system designed for extremely parallel/concurrent file processing and queries that support both metadata and vector search operations.

## Overview

Quipu is built for concurrency testing with files and testing vector database performance under high load. The name was randomly chosen, though it evokes the ancient Incan recording system of knotted strings - fitting for a file indexing system.

Currently implements the vector database API with LanceDB, providing semantic search capabilities across file metadata, filenames, and content embeddings with async/await throughout.

## Features

**Implemented**

- **Vector Database Integration**: Full LanceDB integration with separate tables for metadata, filename embeddings, text content embeddings, and image content embeddings
- **Cross-Platform File Paths**: Safe UTF-8/UTF-16 path handling across Unix and Windows systems
- **Semantic Search**: Query by filename similarity, text content similarity, image content similarity, or combined weighted search
- **Rich Metadata Filtering**: Filter by file type, size, timestamps, permissions, MIME types, tags, and custom conditions
- **Flexible Query Builder**: Fluent API for building complex search queries with multiple filters
- **File Operations**: Add, update, move, and delete files with automatic embedding management
- **Async Throughout**: Full async/await support for concurrent operations

**TODO**

- **Token Chunking**: Implement chunker for processing large text files
- **Benchmarking**: Performance testing to optimize MPSC + worker thread pools
- **Embedder**: Separate embedding modules for text and image content
- **File Preprocessing**: Preprocessor for extracting content from various file formats

## Quick Start

```rust
use quipu::LanceStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // There should ONLY BE ONE INSTANCE of this shared across the app. Otherwise there might be undefined behaviours
    let store = LanceStore::new("data/lance.db").await?;

    // Search by metadata only
    let results = store.query()
        .mime_type("text/plain")
        .size_range(1024, 1024 * 1024) // 1KB to 1MB
        .limit(10)
        .execute()
        .await?;

    // Semantic search by filename
    let query_embedding = vec![0.1; 384]; // Your embedding here
    let results = store.query()
        .by_filename(&query_embedding)
        .extension("rs")
        .limit(5)
        .execute()
        .await?;

    // Combined semantic search with metadata filters
    let results = store.query()
        .combined_with_weights(&query_embedding, 0.3, 0.4, 0.3)
        .has_any_tags(&["important", "todo"])
        .path_contains("src/")
        .limit(15)
        .execute()
        .await?;

    Ok(())
}
```

## Architecture

### Storage Layer

- **Metadata Table**: File system metadata (path, size, timestamps, permissions, tags)
- **Filename Embeddings**: 384-dimensional vectors for filename semantic search
- **Text Content Embeddings**: 384-dimensional vectors for text content semantic search
- **Image Content Embeddings**: 384-dimensional vectors for image content semantic search

### Query System

The query builder supports multiple search modes:

- **Metadata-only**: Fast filtering without vector operations
- **Semantic-only**: Vector similarity search with optional metadata post-filtering
- **Combined**: Weighted combination of filename, text, and image embeddings

### File Path Handling

Cross-platform `FilePath` type safely handles:

- Unix UTF-8 paths stored directly as bytes
- Windows UTF-8/UTF-16 paths with encoding markers (0xFE/0xFF)

## Query Examples

### Metadata Filtering

```rust
// Find large Rust files modified recently
let results = store.query()
    .extension("rs")
    .size_min(10_000)
    .modified_after(1640995200) // After Jan 1, 2022
    .is_readable()
    .limit(20)
    .execute()
    .await?;
```

### Semantic Search

```rust
// Search by filename similarity
let results = store.query()
    .by_filename(&embedding)
    .path_starts_with("/home/user/projects")
    .limit(10)
    .execute()
    .await?;

// Search by text content similarity
let results = store.query()
    .by_text_content(&embedding)
    .mime_type("text/plain")
    .has_tag("documentation")
    .limit(10)
    .execute()
    .await?;
```

### Combined Search

```rust
// Weighted combination of all embedding types
let results = store.query()
    .combined_with_weights(&embedding, 0.2, 0.6, 0.2) // filename, text, image
    .filter("file_size > 1000 AND modified_time > 1640995200")
    .limit(15)
    .execute()
    .await?;
```

## File Operations

### Adding Files

```rust
use quipu::{AddFile, FileType, create_mock_embedding};

let filename_embedding = create_mock_embedding("document.pdf", "filename");
let content_embedding = create_mock_embedding("document content", "content");

let params = AddFile::new(PathBuf::from("document.pdf"), FileType::File, filename_embedding)
    .mime_type("application/pdf")
    .permissions(0o644)
    .file_size(1024)
    .tags(&["important", "work"])
    .content_embedding(content_embedding)
    .content_hash("content_hash");

let file_id = store.add_text_file(&params).await?;
```

### Updating Content

```rust
// Update text file content with new embedding
store.update_text_file_content(
    &file_path,
    "new content",
    new_embedding,
    "new_content_hash"
).await?;
```

## Dependencies

- **LanceDB**: Vector database with Arrow integration
- **Tokio**: Async runtime for concurrent operations
- **Arrow**: Columnar data format
- **Serde**: Serialization
- **rust-bert**: Transformer models (for future embedding generation)

## Development

```bash
# Run tests
cargo test

# Run example
cargo run --example query_usage

# Build
cargo build --release
```

## License

This project is open source. The name "Quipu" was randomly chosen but evokes the ancient Incan system of recording information with knotted strings - a fitting metaphor for a file indexing system.
