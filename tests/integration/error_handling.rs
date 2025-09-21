use crate::helpers::{create_test_document_params, create_test_store};
use quipu::{AddFile, FilePath, FileType, compute_md5_hash, create_mock_embedding};
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::test]
async fn test_error_conditions() {
    let (store, _path) = create_test_store().await;

    let mut params = create_test_document_params();
    params.content_embedding = Some(create_mock_embedding("content", "content"));
    params.content_hash = Some("".to_string());
    let result = store.add_text_file(&params).await;
    assert!(result.is_ok());

    let params = create_test_document_params().override_file_path(PathBuf::from("/path/to/"));
    let result = store.add_text_file(&params).await;
    assert!(result.is_ok());

    let params = create_test_document_params()
        .override_file_path(PathBuf::from(format!("/path/to/{}", "a".repeat(1000))));
    let result = store.add_text_file(&params).await;
    assert!(result.is_ok());

    let mut params = create_test_document_params();
    params.file_size = u64::MAX;
    let result = store.add_text_file(&params).await;
    assert!(result.is_ok());

    let mut params = create_test_document_params();
    params.file_size = 0;
    let result = store.add_text_file(&params).await;
    assert!(result.is_ok());

    let mut params = create_test_document_params();
    params.modified_time = i64::MIN;
    params.accessed_time = Some(i64::MIN);
    params.created_time = Some(i64::MIN);
    let result = store.add_text_file(&params).await;
    assert!(result.is_ok());

    let mut params = create_test_document_params();
    params.modified_time = i64::MAX;
    params.accessed_time = Some(i64::MAX);
    params.created_time = Some(i64::MAX);
    let result = store.add_text_file(&params).await;
    assert!(result.is_ok());

    let mut params = create_test_document_params();
    params.tags = Some(vec![]);
    let result = store.add_text_file(&params).await;
    assert!(result.is_ok());

    let mut params = create_test_document_params();
    params.tags = Some(vec!["a".repeat(1000)]);
    let result = store.add_text_file(&params).await;
    assert!(result.is_ok());

    let mut params = create_test_document_params();
    params.content_hash = Some("a".repeat(1000));
    let result = store.add_text_file(&params).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_error_handling_edge_cases() {
    let (store, _temp_dir) = create_test_store().await;

    // Test move file with non-existent source
    let result = store
        .move_file(&PathBuf::from("/non/existent"), PathBuf::from("/new/path"))
        .await;
    assert!(result.is_err());

    // Test update text file content with non-existent file
    let result = store
        .update_text_file_content(
            &PathBuf::from("/non/existent"),
            "content",
            vec![0.0; 384],
            "hash",
        )
        .await;
    assert!(result.is_err());

    // Test delete file with non-existent file (should succeed silently)
    let result = store.delete_file(&PathBuf::from("/non/existent")).await;
    assert!(result.is_ok());

    // Test reindex file with non-existent file
    let result = store.reindex_file(&PathBuf::from("/non/existent")).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_concurrent_operations() {
    let (store, temp_dir) = create_test_store().await;
    let temp_path = temp_dir.path();

    // Wrap the store in Arc to share across concurrent tasks
    let store = Arc::new(store);

    // Test concurrent file operations
    let mut handles = vec![];

    for i in 0..5 {
        let temp_path_str = temp_path.to_string_lossy().to_string();
        let store_clone = Arc::clone(&store);

        let handle = tokio::spawn(async move {
            let filename_embedding =
                create_mock_embedding(&format!("concurrent_file_{}.txt", i), "filename");
            let params = AddFile::new(
                PathBuf::from(format!("{}/concurrent_file_{}.txt", temp_path_str, i)),
                FileType::File,
                filename_embedding,
            )
            .mime_type("text/plain")
            .file_size(1024)
            .permissions(0o644);

            store_clone.add_text_file(&params).await
        });

        handles.push(handle);
    }

    // Wait for all concurrent operations to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    // Verify all files were created using the shared store
    for i in 0..5 {
        let file_path = format!("{}/concurrent_file_{}.txt", temp_path.to_string_lossy(), i);
        let metadata = store
            .get_metadata_by_path(&PathBuf::from(file_path))
            .await
            .unwrap();
        assert!(metadata.is_some());
    }
}

#[tokio::test]
async fn test_large_content_handling() {
    let (store, _path) = create_test_store().await;

    // Test with very large content
    let large_content = "x".repeat(100_000); // 100KB content
    let filename_embedding = create_mock_embedding("large_file.txt", "filename");
    let content_embedding = create_mock_embedding(&large_content, "content");

    let params = AddFile::new(
        PathBuf::from("/path/to/large_file.txt"),
        FileType::File,
        filename_embedding,
    )
    .content_embedding(content_embedding)
    .content_hash(&large_content)
    .mime_type("text/plain")
    .file_size(large_content.len() as u64)
    .permissions(0o644);

    let file_id = store.add_text_file(&params).await.unwrap();

    let metadata = store.get_metadata(&file_id).await.unwrap().unwrap();
    assert_eq!(metadata.file_size, large_content.len() as u64);
    assert_eq!(metadata.filename(), Some("large_file.txt".to_string()));
}

#[tokio::test]
async fn test_special_file_paths() {
    let (store, _path) = create_test_store().await;

    let special_paths = vec![
        "/path/with spaces/file.txt",
        "/path/with-dashes/file.txt",
        "/path/with_underscores/file.txt",
        "/path/with.dots/file.txt",
        "/path/with@symbols/file.txt",
        "/path/with#hash/file.txt",
        "/path/with$variable/file.txt",
        "/path/with&and/file.txt",
        "/path/with*asterisk/file.txt",
        "/path/with(parentheses)/file.txt",
        "/path/with[brackets]/file.txt",
        "/path/with{braces}/file.txt",
        "/path/with|pipe/file.txt",
        "/path/with\\backslash/file.txt",
        "/path/with;colon/file.txt",
        "/path/with'quote/file.txt",
        "/path/with\"doublequote/file.txt",
        "/path/with<less/file.txt",
        "/path/with>greater/file.txt",
    ];

    for (i, path) in special_paths.iter().enumerate() {
        let filename_embedding = create_mock_embedding(&format!("file_{}.txt", i), "filename");
        let params = AddFile::new(PathBuf::from(path), FileType::File, filename_embedding)
            .mime_type("text/plain")
            .file_size(1024)
            .permissions(0o644);

        let file_id = store.add_text_file(&params).await.unwrap();

        let metadata = store.get_metadata(&file_id).await.unwrap().unwrap();
        assert_eq!(metadata.file_path, FilePath::from(PathBuf::from(*path)));
    }
}

#[tokio::test]
async fn test_extreme_tag_values() {
    let (store, _path) = create_test_store().await;

    // Test with extremely long tags
    let long_tag = "a".repeat(1000);
    let many_tags: Vec<String> = vec!["tag".to_string(); 1000]; // 1000 identical tags

    let filename_embedding = create_mock_embedding("extreme_tags.txt", "filename");

    let params = AddFile::new(
        PathBuf::from("/path/to/extreme_tags.txt"),
        FileType::File,
        filename_embedding,
    )
    .mime_type("text/plain")
    .tags(&[&long_tag])
    .file_size(1024)
    .permissions(0o644);

    let file_id = store.add_text_file(&params).await.unwrap();
    let metadata = store.get_metadata(&file_id).await.unwrap().unwrap();
    assert_eq!(metadata.tags, Some(vec![long_tag.clone()]));

    // Test with many tags
    let filename_embedding2 = create_mock_embedding("many_tags.txt", "filename");
    let tag_refs: Vec<&str> = many_tags.iter().map(|s| s.as_str()).collect();
    let params2 = AddFile::new(
        PathBuf::from("/path/to/many_tags.txt"),
        FileType::File,
        filename_embedding2,
    )
    .mime_type("text/plain")
    .tags(&tag_refs)
    .file_size(1024)
    .permissions(0o644);

    let file_id2 = store.add_text_file(&params2).await.unwrap();
    let metadata2 = store.get_metadata(&file_id2).await.unwrap().unwrap();
    assert_eq!(metadata2.tags, Some(many_tags.clone()));
}

#[tokio::test]
async fn test_search_error_scenarios() {
    let (store, _path) = create_test_store().await;

    let empty_embedding = vec![];
    let result = store
        .query()
        .by_filename(&empty_embedding)
        .limit(5)
        .execute()
        .await;
    assert!(result.is_err());

    let query_embedding = create_mock_embedding("test", "filename");
    let result = store
        .query()
        .by_filename(&query_embedding)
        .limit(1_000_000)
        .execute()
        .await;
    assert!(result.is_ok());

    let result = store
        .query()
        .filter("nonexistent_field = 'value'")
        .limit(5)
        .execute()
        .await;
    match &result {
        Ok(results) => println!(
            "Search with nonexistent field returned OK with {} results",
            results.len()
        ),
        Err(e) => println!("Search with nonexistent field returned error: {:?}", e),
    }
    assert!(result.is_err()); // LanceDB correctly throws an error for nonexistent fields

    let result = store
        .query()
        .filter("id = 'malformed\t\n'")
        .limit(5)
        .execute()
        .await;
    match &result {
        Ok(results) => println!(
            "Search with malformed value returned OK with {} results",
            results.len()
        ),
        Err(e) => println!("Search with malformed value returned error: {:?}", e),
    }
    assert!(result.is_ok()); // Malformed values are handled gracefully by LanceDB
}

#[tokio::test]
async fn test_file_operation_error_scenarios() {
    let (store, temp_dir) = create_test_store().await;
    let temp_path = temp_dir.path();

    let mut params = create_test_document_params();
    params.file_path = PathBuf::from("").into();
    let result = store.add_text_file(&params).await;
    assert!(result.is_ok());

    let long_path = format!(
        "/extremely/long/path/that/exceeds/filesystem/limits/{}",
        "a".repeat(1000)
    );
    let mut params = create_test_document_params();
    params.file_path = PathBuf::from(long_path).into();
    let result = store.add_text_file(&params).await;
    assert!(result.is_ok());

    let nonexistent_path = temp_path.join("nonexistent.txt");
    let result = store
        .update_text_file_content(&nonexistent_path, "content", vec![0.0; 384], "hash")
        .await;
    assert!(result.is_err());

    let valid_file = temp_path.join("valid_file.txt");
    std::fs::write(&valid_file, "content").unwrap();

    let params = create_test_document_params().override_file_path(valid_file.clone());
    let _file_id = store.add_text_file(&params).await.unwrap();

    let invalid_dest = PathBuf::from("");
    let result = store.move_file(&valid_file, invalid_dest).await;
    assert!(result.is_ok()); // move_file only updates database metadata, doesn't validate destination path
}

#[tokio::test]
async fn test_metadata_operation_edge_cases() {
    let (store, _path) = create_test_store().await;

    let result = store.get_metadata("").await.unwrap();
    assert!(result.is_none());

    let long_id = "a".repeat(10000);
    let result = store.get_metadata(&long_id).await.unwrap();
    assert!(result.is_none());

    let special_path = "/path/with/special/characters/!@#$%^&*()/file.txt";
    let mut params = create_test_document_params();
    params.file_path = PathBuf::from(special_path).into();
    let file_id = store.add_text_file(&params).await.unwrap();

    let result = store
        .get_metadata_by_path(&PathBuf::from(special_path))
        .await
        .unwrap();
    assert!(result.is_some());
    assert_eq!(result.unwrap().id, file_id);

    let cleanup_count = store.cleanup_missing_files().await.unwrap();
    assert_eq!(cleanup_count, 1);
}

#[tokio::test]
async fn test_concurrent_search_operations() {
    let (store, temp_dir) = create_test_store().await;
    let temp_path = temp_dir.path();

    let mut file_ids = vec![];
    for i in 0..10 {
        let file_path = temp_path.join(format!("concurrent_search_{}.txt", i));
        let content = format!("Concurrent search test file {}", i);
        std::fs::write(&file_path, &content).unwrap();

        let mut params = create_test_document_params().override_file_path(file_path);
        params.content_embedding = Some(create_mock_embedding(&content, "content"));
        params.content_hash = Some(compute_md5_hash(&content));

        let file_id = store.add_text_file(&params).await.unwrap();
        file_ids.push(file_id);
    }

    let mut handles = vec![];

    for _i in 0..5 {
        let store_clone = Arc::new(store.clone());
        let handle = tokio::spawn(async move {
            let query_embedding = create_mock_embedding("concurrent search", "content");
            let results = store_clone
                .query()
                .by_text_content(&query_embedding)
                .limit(10)
                .execute()
                .await
                .unwrap();
            results.len()
        });
        handles.push(handle);
    }

    for handle in handles {
        let result_count = handle.await.unwrap();
        assert!(result_count > 0);
    }
}

#[tokio::test]
async fn test_memory_usage_with_many_files() {
    let (store, temp_dir) = create_test_store().await;
    let temp_path = temp_dir.path();

    let mut file_ids = vec![];
    for i in 0..100 {
        let file_path = temp_path.join(format!("memory_test_{:03}.txt", i));
        let content = format!("Memory test file number {}", i);
        std::fs::write(&file_path, &content).unwrap();

        let mut params = create_test_document_params().override_file_path(file_path);
        params.filename_embedding =
            create_mock_embedding(&format!("memory file {}", i), "filename");

        let file_id = store.add_text_file(&params).await.unwrap();
        file_ids.push(file_id);
    }

    for file_id in &file_ids {
        let metadata = store.get_metadata(file_id).await.unwrap();
        assert!(metadata.is_some());
    }

    let search_embedding = create_mock_embedding("memory test", "filename");
    let results = store
        .query()
        .by_filename(&search_embedding)
        .limit(50)
        .execute()
        .await
        .unwrap();

    assert!(!results.is_empty());
    assert!(results.len() <= 50);
}
