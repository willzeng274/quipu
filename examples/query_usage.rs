
use quipu::LanceStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = LanceStore::new("data/lance.db").await?;
    let query_embedding = vec![0.1; 384];

    let results = store.query()
        .mime_type("text/plain")
        .limit(10)
        .execute()
        .await?;
    
    println!("Found {} text files:", results.len());
    for file in &results {
        println!("  {} ({} bytes)", file.file_path_lossy, file.file_size);
    }

    let results = store.query()
        .by_filename(&query_embedding)
        .limit(5)
        .execute()
        .await?;
    
    println!("\nSemantic filename search results:");
    for file in &results {
        if let Some(filename) = file.filename() {
            println!("  {}: {}", filename, file.file_path_lossy);
        }
    }

    let results = store.query()
        .by_text_content(&query_embedding)
        .mime_type("text/plain")
        .path_contains("src/")
        .limit(10)
        .execute()
        .await?;
    
    println!("\nText content search in src/ directory:");
    for file in &results {
        println!("  {} ({})", file.file_path_lossy, 
                 file.mime_type.as_deref().unwrap_or("unknown"));
    }

    let results = store.query()
        .combined_with_weights(&query_embedding, 0.2, 0.6, 0.2)
        .has_any_tags(&["important", "todo"])
        .size_range(1024, 1024 * 1024) // 1KB to 1MB
        .limit(15)
        .execute()
        .await?;
    
    println!("\nCombined search with tags and size filters:");
    for file in &results {
        let tags = file.tags.as_ref()
            .map(|t| t.join(", "))
            .unwrap_or_else(|| "no tags".to_string());
        println!("  {} - {} bytes, tags: [{}]", 
                 file.file_path_lossy, file.file_size, tags);
    }

    let results = store.query()
        .by_text_content(&query_embedding)
        .extension("rs")
        .modified_after(1640995200) // After Jan 1, 2022
        .is_readable()
        .path_starts_with("/home/user/projects")
        .limit(20)
        .execute()
        .await?;
    
    println!("\nRust files modified after 2022:");
    for file in &results {
        let modified = chrono::DateTime::from_timestamp(file.modified_time, 0)
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| "unknown".to_string());
        println!("  {} - modified: {}", file.file_path_lossy, modified);
    }

    let results = store.query()
        .filter("file_size > 1000 AND modified_time > 1640995200")
        .mime_type("application/json")
        .has_tag("config")
        .limit(50)
        .execute()
        .await?;
    
    println!("\nConfig JSON files > 1KB modified after 2022:");
    for file in &results {
        let size_kb = file.file_size as f64 / 1024.0;
        println!("  {} - {:.1} KB", file.file_path_lossy, size_kb);
    }

    Ok(())
}