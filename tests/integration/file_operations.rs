use quipu::{
    AddFile, FilePath, FileType,
    add_directory, scan_directory, create_mock_embedding, compute_md5_hash,
};
use std::path::PathBuf;
use crate::helpers::{
    create_test_store,
    create_test_document_params,
    create_test_directory_params,
    create_test_image_params,
};

#[tokio::test]
async fn test_add_file_basic() {
    let (store, _path) = create_test_store().await;

    let file_id = store
        .add_text_file(&create_test_document_params())
        .await
        .unwrap();

    let metadata = store.get_metadata(&file_id).await.unwrap();
    assert!(metadata.is_some());

    let metadata = metadata.unwrap();
    assert_eq!(PathBuf::from(metadata.file_path), PathBuf::from("/path/to/test_document.txt"));
    assert_eq!(metadata.file_type, FileType::File);
}

#[tokio::test]
async fn test_add_file_with_content() {
    let (store, _path) = create_test_store().await;

    let filename_embedding = create_mock_embedding("content_doc.txt", "filename");
    let content_embedding =
        create_mock_embedding("This is test content for the document.", "content");

    let params = AddFile::new(PathBuf::from("/path/to/content_doc.txt"), FileType::File, filename_embedding.clone())
        .content_embedding(content_embedding.clone())
        .content_hash("This is test content for the document.")
        .mime_type("text/plain")
        .tags(&["content", "text"])
        .file_size(2048)
        .permissions(0o644);
    let file_id = store.add_text_file(&params).await.unwrap();

    let metadata = store.get_metadata(&file_id).await.unwrap();
    assert!(metadata.is_some());

    let metadata = metadata.unwrap();
    assert_eq!(metadata.filename(), Some("content_doc.txt".to_string()));
}

#[tokio::test]
async fn test_get_nonexistent_metadata() {
    let (store, _path) = create_test_store().await;

    let result = store.get_metadata("nonexistent-id").await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_file_type_validation() {
    let (store, _path) = create_test_store().await;

    let content_embedding = create_mock_embedding("Directory content that should fail", "content");

    let mut params = create_test_directory_params();
    params.content_embedding = Some(content_embedding.clone());
    params.content_hash = Some(compute_md5_hash("Directory content that should fail"));
    let result = store.add_text_file(&params).await;
    assert!(result.is_err());

    let mut params = AddFile::new(PathBuf::from("/path/to/test_link"), FileType::Symlink, create_mock_embedding("test_link", "filename"))
        .tags(&["test", "symlink"])
        .permissions(0o777);
    params.content_embedding = Some(content_embedding.clone());
    params.content_hash = Some(compute_md5_hash("Symlink content that should fail"));
    let result = store.add_text_file(&params).await;
    assert!(result.is_err());

    let result = store.add_text_file(&create_test_directory_params()).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_image_file_support() {
    let (store, _path) = create_test_store().await;

    let file_id = store
        .add_image_file(&create_test_image_params())
        .await
        .unwrap();

    let dir_id = store
        .add_image_file(&create_test_directory_params())
        .await
        .unwrap();

    let query_embedding = create_mock_embedding("image content", "content");
    let results = store.query()
        .by_image_content(&query_embedding)
        .limit(5)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);

    let file_metadata = store.get_metadata(&file_id).await.unwrap().unwrap();
    assert_eq!(file_metadata.filename(), Some("test_image.png".to_string()));
    assert_eq!(file_metadata.file_type, FileType::File);

    let dir_metadata = store.get_metadata(&dir_id).await.unwrap().unwrap();
    assert_eq!(dir_metadata.filename(), Some("test_dir".to_string()));
    assert_eq!(dir_metadata.file_type, FileType::Directory);
}

#[tokio::test]
async fn test_delete_file_operation() {
    let (store, temp_dir) = create_test_store().await;

    let temp_path = temp_dir.path();
    let file_path = temp_path.join("file_to_delete.txt");
    std::fs::write(&file_path, "content to delete").unwrap();

    let params = create_test_document_params().override_file_path(file_path.clone());

    let _file_id = store.add_text_file(&params).await.unwrap();

    let metadata_before = store.get_metadata_by_path(&file_path).await.unwrap();
    assert!(metadata_before.is_some());

    store.delete_file(&file_path).await.unwrap();

    let metadata_after = store.get_metadata_by_path(&file_path).await.unwrap();
    assert!(metadata_after.is_none());
}

#[tokio::test]
async fn test_file_move_operation() {
    let (store, temp_dir) = create_test_store().await;

    let temp_path = temp_dir.path();
    let original_path = temp_path.join("original_file.txt");
    let new_path = temp_path.join("moved_file.txt");

    std::fs::write(&original_path, "test content").unwrap();

    let params = create_test_document_params().override_file_path(original_path.clone());

    let _file_id = store.add_text_file(&params).await.unwrap();

    store.move_file(
        &original_path,
        new_path.clone()
    ).await.unwrap();

    let metadata_after = store.get_metadata_by_path(&new_path).await.unwrap();
    assert!(metadata_after.is_some());
    assert_eq!(metadata_after.unwrap().filename(), Some("moved_file.txt".to_string()));

    let old_metadata = store.get_metadata_by_path(&original_path).await.unwrap();
    assert!(old_metadata.is_none());
}

#[tokio::test]
async fn test_basic_crud_operations() {
    let (store, temp_dir) = create_test_store().await;

    let temp_path = temp_dir.path();
    let file_path = temp_path.join("basic_crud_test.txt");
    std::fs::write(&file_path, "test content").unwrap();

    let mut params = create_test_document_params().override_file_path(file_path.clone());
    params.content_embedding = Some(create_mock_embedding("test content", "content"));
    params.content_hash = Some(compute_md5_hash("test content"));

    let _file_id = store.add_text_file(&params).await.unwrap();

    let metadata = store.get_metadata_by_path(&file_path).await.unwrap();
    assert!(metadata.is_some());
    assert_eq!(metadata.unwrap().filename(), Some("basic_crud_test.txt".to_string()));

    store.update_tags(&file_path, vec!["updated".to_string()]).await.unwrap();

    let updated_metadata = store.get_metadata_by_path(&file_path).await.unwrap();
    assert!(updated_metadata.is_some());
    assert_eq!(updated_metadata.unwrap().tags, Some(vec!["updated".to_string()]));

    store.delete_file(&file_path).await.unwrap();

    let deleted_metadata = store.get_metadata_by_path(&file_path).await.unwrap();
    assert!(deleted_metadata.is_none());
}

#[tokio::test]
async fn test_update_text_file_content() {
    let (store, temp_dir) = create_test_store().await;

    let temp_path = temp_dir.path();
    let file_path = temp_path.join("content_file.txt");
    let original_content = "Original content";
    let new_content = "Updated content with more text";

    std::fs::write(&file_path, original_content).unwrap();

    let mut params = create_test_document_params().override_file_path(file_path.clone());
    params.file_size = original_content.len() as u64;
    params.content_embedding = Some(create_mock_embedding(original_content, "content"));
    params.content_hash = Some(compute_md5_hash(original_content));

    let _file_id = store.add_text_file(&params).await.unwrap();

    let new_embedding = create_mock_embedding(new_content, "content");
    let new_content_hash = compute_md5_hash(new_content);

    store.update_text_file_content(
        &file_path,
        new_content,
        new_embedding.clone(),
        &new_content_hash
    ).await.unwrap();

    let metadata_after = store.get_metadata_by_path(&file_path).await.unwrap();
    assert!(metadata_after.is_some());
    let metadata = metadata_after.unwrap();
    assert_eq!(metadata.file_size, new_content.len() as u64);

    let search_results = store.query()
        .by_text_content(&new_embedding)
        .limit(5)
        .execute()
        .await
        .unwrap();

    assert!(!search_results.is_empty(), "Search should find the updated content");
    assert_eq!(search_results.len(), 1, "Should find exactly one result for the updated content");

    let found_file = &search_results[0];
    assert_eq!(PathBuf::from(found_file.file_path.clone()), file_path);
    assert_eq!(found_file.file_size, new_content.len() as u64);
}

#[tokio::test]
async fn test_update_image_file_content() {
    let (store, temp_dir) = create_test_store().await;

    let temp_path = temp_dir.path();
    let file_path = temp_path.join("test_image.png");

    let png_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    std::fs::write(&file_path, &png_data).unwrap();

    let mut params = create_test_image_params().override_file_path(file_path.clone());
    params.file_size = png_data.len() as u64;

    let _file_id = store.add_image_file(&params).await.unwrap();

    let new_image_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D];
    let new_embedding = create_mock_embedding("new image content", "content");
    let new_content_hash = compute_md5_hash(&String::from_utf8_lossy(&new_image_data));

    store.update_image_file_content(
        &file_path,
        &new_image_data,
        new_embedding,
        &new_content_hash
    ).await.unwrap();

    let metadata_after = store.get_metadata_by_path(&file_path).await.unwrap();
    assert!(metadata_after.is_some());
    let metadata = metadata_after.unwrap();
    assert_eq!(metadata.file_size, new_image_data.len() as u64);
}

#[tokio::test]
async fn test_reindex_file_operation() {
    let (store, temp_dir) = create_test_store().await;

    let temp_path = temp_dir.path();
    let file_path = temp_path.join("file_to_reindex.txt");
    let original_content = "original content";

    std::fs::write(&file_path, original_content).unwrap();

    let mut params = create_test_document_params().override_file_path(file_path.clone());
    params.file_size = original_content.len() as u64;

    let _file_id = store.add_text_file(&params).await.unwrap();

    std::thread::sleep(std::time::Duration::from_millis(100));

    let new_content = "modified content";
    std::fs::write(&file_path, new_content).unwrap();

    store.reindex_file(&file_path).await.unwrap();

    let metadata_after = store.get_metadata_by_path(&file_path).await.unwrap();
    assert!(metadata_after.is_some());
    let metadata = metadata_after.unwrap();
    assert_eq!(metadata.file_size, new_content.len() as u64);
}

#[tokio::test]
async fn test_add_directory_operation() {
    let (store, temp_dir) = create_test_store().await;

    let temp_path = temp_dir.path();
    let dir_path = temp_path.join("test_directory");
    std::fs::create_dir(&dir_path).unwrap();

    let filename_embedding = create_mock_embedding("test_directory", "filename");

    let _dir_id = add_directory(&store, &dir_path, filename_embedding).await.unwrap();

    let metadata = store.get_metadata_by_path(&dir_path).await.unwrap();
    assert!(metadata.is_some());
    let metadata = metadata.unwrap();
    assert_eq!(metadata.file_type, FileType::Directory);
    assert_eq!(metadata.file_size, 0);
}

#[tokio::test]
async fn test_scan_directory_operation() {
    let (store, temp_dir) = create_test_store().await;

    let temp_path = temp_dir.path();
    let dir_path = temp_path.join("scan_test_dir");
    std::fs::create_dir(&dir_path).unwrap();

    let file1_path = dir_path.join("file1.txt");
    let file2_path = dir_path.join("file2.txt");
    let file3_path = dir_path.join("file3.txt");

    std::fs::write(&file1_path, "content 1").unwrap();
    std::fs::write(&file2_path, "content 2").unwrap();
    std::fs::write(&file3_path, "content 3").unwrap();

    let file_ids = scan_directory(&store, &dir_path).await.unwrap();

    assert_eq!(file_ids.len(), 3);

    for file_id in &file_ids {
        let metadata = store.get_metadata(file_id).await.unwrap();
        assert!(metadata.is_some());
    }
}

#[tokio::test]
async fn test_update_tags_operation() {
    let (store, temp_dir) = create_test_store().await;

    let temp_path = temp_dir.path();
    let file_path = temp_path.join("tagged_file.txt");
    std::fs::write(&file_path, "content").unwrap();

    let mut params = create_test_document_params().override_file_path(file_path.clone());
    params.tags = Some(vec!["original".to_string(), "test".to_string()]);

    let file_id = store.add_text_file(&params).await.unwrap();

    let new_tags = vec!["updated".to_string(), "production".to_string(), "important".to_string()];
    store.update_tags(&file_path, new_tags.clone()).await.unwrap();

    let metadata_after = store.get_metadata_by_path(&file_path).await.unwrap();
    assert!(metadata_after.is_some());
    let metadata = metadata_after.unwrap();
    assert_eq!(metadata.tags, Some(new_tags));

    let search_results = store.query()
        .has_all_tags(&["important"])
        .limit(5)
        .execute()
        .await
        .unwrap();
    assert!(!search_results.is_empty());
    assert_eq!(search_results[0].id, file_id);
}

#[tokio::test]
async fn test_get_metadata_by_path() {
    let (store, temp_dir) = create_test_store().await;

    let temp_path = temp_dir.path();
    let file_path = temp_path.join("path_test_file.txt");
    std::fs::write(&file_path, "content").unwrap();

    let params = create_test_document_params().override_file_path(file_path.clone());

    let file_id = store.add_text_file(&params).await.unwrap();

    let metadata = store.get_metadata_by_path(&file_path).await.unwrap();
    assert!(metadata.is_some());
    let metadata = metadata.unwrap();
    assert_eq!(metadata.id, file_id);
    assert_eq!(metadata.filename(), Some("path_test_file.txt".to_string()));

    let original_path_bytes = FilePath::from(file_path.clone());
    assert_eq!(original_path_bytes.bytes, metadata.file_path.bytes);

    let non_existent_metadata = store.get_metadata_by_path(&PathBuf::from("/non/existent/path")).await.unwrap();
    assert!(non_existent_metadata.is_none());
}

#[tokio::test]
async fn test_file_lifecycle_workflow() {
    let (store, temp_dir) = create_test_store().await;

    let temp_path = temp_dir.path();
    let file_path = temp_path.join("lifecycle_file.txt");
    let content = "initial content";
    std::fs::write(&file_path, content).unwrap();

    let mut params = create_test_document_params().override_file_path(file_path.clone());
    params.tags = Some(vec!["workflow".to_string()]);

    let _file_id = store.add_text_file(&params).await.unwrap();

    let metadata1 = store.get_metadata_by_path(&file_path).await.unwrap();
    assert!(metadata1.is_some());

    let new_content = "updated content";
    let new_embedding = create_mock_embedding(new_content, "content");
    let new_content_hash = compute_md5_hash(new_content);
    store.update_text_file_content(
        &file_path,
        new_content,
        new_embedding,
        &new_content_hash
    ).await.unwrap();

    let metadata2 = store.get_metadata_by_path(&file_path).await.unwrap();
    assert!(metadata2.is_some());
    assert_eq!(metadata2.unwrap().file_size, new_content.len() as u64);

    store.update_tags(&file_path, vec!["workflow".to_string(), "updated".to_string()]).await.unwrap();

    let new_file_path = temp_path.join("moved_lifecycle_file.txt");
    store.move_file(
        &file_path,
        new_file_path.clone()
    ).await.unwrap();

    let metadata3 = store.get_metadata_by_path(&new_file_path).await.unwrap();
    assert!(metadata3.is_some());
    assert_eq!(metadata3.unwrap().filename(), Some("moved_lifecycle_file.txt".to_string()));

    store.reindex_file(&new_file_path).await.unwrap();

    store.delete_file(&new_file_path).await.unwrap();

    let metadata4 = store.get_metadata_by_path(&new_file_path).await.unwrap();
    assert!(metadata4.is_none());
}

#[tokio::test]
async fn test_batch_file_operations() {
    let (store, temp_dir) = create_test_store().await;
    let temp_path = temp_dir.path();

    let mut file_ids = vec![];
    let mut file_paths = vec![];

    for i in 0..5 {
        let file_path = temp_path.join(format!("batch_file_{}.txt", i));
        let content = format!("Batch file {} content", i);
        std::fs::write(&file_path, &content).unwrap();

        let mut params = create_test_document_params().override_file_path(file_path.clone());
        params.content_embedding = Some(create_mock_embedding(&content, "content"));
        params.content_hash = Some(compute_md5_hash(&content));

        let file_id = store.add_text_file(&params).await.unwrap();
        file_ids.push(file_id);
        file_paths.push(file_path);
    }

    assert_eq!(file_ids.len(), 5);
    for file_id in &file_ids {
        let metadata = store.get_metadata(file_id).await.unwrap();
        assert!(metadata.is_some());
    }

    for file_path in &file_paths {
        store.delete_file(file_path).await.unwrap();
    }

    for file_id in &file_ids {
        let metadata = store.get_metadata(file_id).await.unwrap();
        assert!(metadata.is_none());
    }
}

#[tokio::test]
async fn test_file_search_after_operations() {
    let (store, temp_dir) = create_test_store().await;
    let temp_path = temp_dir.path();

    let file_path = temp_path.join("search_test.txt");
    let content = "This is searchable content for testing";
    std::fs::write(&file_path, content).unwrap();

    let mut params = create_test_document_params().override_file_path(file_path.clone());
    params.content_embedding = Some(create_mock_embedding(content, "content"));
    params.content_hash = Some(compute_md5_hash(content));

    let file_id = store.add_text_file(&params).await.unwrap();

    let search_embedding = create_mock_embedding("searchable content", "content");
    let search_results = store.query()
        .by_text_content(&search_embedding)
        .limit(10)
        .execute()
        .await
        .unwrap();

    assert_eq!(search_results.len(), 1);
    assert_eq!(search_results[0].id, file_id);

    let new_content = "Updated searchable content";
    let new_embedding = create_mock_embedding(new_content, "content");
    let new_hash = compute_md5_hash(new_content);

    store.update_text_file_content(
        &file_path,
        new_content,
        new_embedding.clone(),
        &new_hash
    ).await.unwrap();

    let updated_search_results = store.query()
        .by_text_content(&new_embedding)
        .limit(10)
        .execute()
        .await
        .unwrap();

    assert_eq!(updated_search_results.len(), 1);
    assert_eq!(updated_search_results[0].id, file_id);
    assert_eq!(updated_search_results[0].file_size, new_content.len() as u64);
}

#[tokio::test]
async fn test_directory_with_nested_files() {
    let (store, temp_dir) = create_test_store().await;
    let temp_path = temp_dir.path();

    let nested_dir = temp_path.join("parent").join("child");
    std::fs::create_dir_all(&nested_dir).unwrap();

    let file1_path = nested_dir.join("nested_file1.txt");
    let file2_path = nested_dir.join("nested_file2.txt");

    let content1 = "Nested file 1 content";
    let content2 = "Nested file 2 content";

    std::fs::write(&file1_path, content1).unwrap();
    std::fs::write(&file2_path, content2).unwrap();

    let mut params1 = create_test_document_params().override_file_path(file1_path.clone());
    params1.content_embedding = Some(create_mock_embedding(content1, "content"));
    params1.content_hash = Some(compute_md5_hash(content1));

    let mut params2 = create_test_document_params().override_file_path(file2_path.clone());
    params2.content_embedding = Some(create_mock_embedding(content2, "content"));
    params2.content_hash = Some(compute_md5_hash(content2));

    let file_id1 = store.add_text_file(&params1).await.unwrap();
    let file_id2 = store.add_text_file(&params2).await.unwrap();

    let search_results = store.query()
        .path_contains("parent/child")
        .limit(10)
        .execute()
        .await
        .unwrap();

    assert_eq!(search_results.len(), 2);

    let found_ids: std::collections::HashSet<_> = search_results.iter().map(|r| r.id.clone()).collect();
    assert!(found_ids.contains(&file_id1));
    assert!(found_ids.contains(&file_id2));
}

#[tokio::test]
async fn test_large_file_support() {
    let (store, temp_dir) = create_test_store().await;
    let temp_path = temp_dir.path();

    let large_file_path = temp_path.join("large_file.txt");
    let large_content = "x".repeat(1_000_000);
    std::fs::write(&large_file_path, &large_content).unwrap();

    let mut params = create_test_document_params().override_file_path(large_file_path.clone());
    params.content_embedding = Some(create_mock_embedding("large file content", "content"));
    params.content_hash = Some(compute_md5_hash(&large_content));
    params.file_size = large_content.len() as u64;

    let file_id = store.add_text_file(&params).await.unwrap();

    let metadata = store.get_metadata(&file_id).await.unwrap().unwrap();
    assert_eq!(metadata.file_size, 1_000_000);
    assert_eq!(metadata.filename(), Some("large_file.txt".to_string()));

    let search_embedding = create_mock_embedding("large file", "filename");
    let search_results = store.query()
        .by_filename(&search_embedding)
        .limit(5)
        .execute()
        .await
        .unwrap();

    assert_eq!(search_results.len(), 1);
    assert_eq!(search_results[0].id, file_id);
}

#[tokio::test]
async fn test_file_permissions_comprehensive() {
    let (store, temp_dir) = create_test_store().await;
    let temp_path = temp_dir.path();

    let permission_tests = vec![
        (0o644, "read_write"),
        (0o755, "execute"),
        (0o600, "private"),
        (0o777, "full_access"),
        (0o444, "read_only"),
    ];

    let mut file_ids = vec![];

    for (permissions, test_name) in &permission_tests {
        let file_path = temp_path.join(format!("{}_file.txt", test_name));
        let content = format!("Content for {} test", test_name);
        std::fs::write(&file_path, &content).unwrap();

        let mut params = create_test_document_params().override_file_path(file_path);
        params.permissions = Some(*permissions);

        let file_id = store.add_text_file(&params).await.unwrap();
        file_ids.push(file_id);
    }

    for (i, file_id) in file_ids.iter().enumerate() {
        let metadata = store.get_metadata(file_id).await.unwrap().unwrap();
        let expected_permissions = permission_tests[i].0;
        assert_eq!(metadata.permissions, Some(expected_permissions));
    }
}
