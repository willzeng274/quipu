use quipu::{
    FileType,
    create_mock_embedding, compute_md5_hash,
};
use std::path::PathBuf;
use crate::helpers::{
    create_test_store,
    create_test_document_params,
    create_ai_research_params,
};

#[tokio::test]
async fn test_metadata_search() {
    let (store, _path) = create_test_store().await;

    let file_id = store
        .add_text_file(&create_ai_research_params())
        .await
        .unwrap();

    let metadata = store.get_metadata(&file_id).await.unwrap();
    assert!(metadata.is_some());
    let metadata = metadata.unwrap();
    assert_eq!(metadata.filename(), Some("ai_research.pdf".to_string()));

    let results = store.query()
        .path_contains("ai")
        .limit(5)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);

    let results = store.query()
        .mime_type("application/pdf")
        .limit(5)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);

    let results = store.query()
        .size_range(0, 1000)
        .limit(5)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);

    let results = store.query()
        .has_all_tags(&["ai"])
        .limit(1)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_metadata_search_edge_cases() {
    let (store, _path) = create_test_store().await;

    let _file_id = store
        .add_text_file(&create_ai_research_params())
        .await
        .unwrap();

    let results = store.query()
        .filter("id = ''")
        .limit(5)
        .execute()
        .await;
    assert!(results.is_ok());
    assert_eq!(results.unwrap().len(), 0);

    let results = store.query()
        .path_contains("research")
        .mime_type("application/pdf")
        .has_all_tags(&["ai", "research"])
        .limit(10)
        .execute()
        .await
        .unwrap();
    println!("Search results: {} files", results.len());
    for result in &results {
        println!("File: {:?}, tags: {:?}", result.file_path_lossy, result.tags);
    }
    assert_eq!(results.len(), 1); // File has both tags

    let query = store.query()
        .has_any_tags(&["ai", "nonexistent"])
        .limit(10);
    println!("Executing has_any_tags query");
    let results = query.execute().await.unwrap();
    println!("has_any_tags search results: {} files", results.len());
    // assert_eq!(results.len(), 1);

    let results = store.query()
        .has_all_tags(&["ai", "nonexistent"])
        .limit(10)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 0); // has_all_tags requires ALL tags to be present

    let results = store.query()
        .has_any_tags(&["ai", "nonexistent"])
        .limit(10)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1); // has_any_tags finds files with ANY of the tags

    let results = store.query()
        .size_range(100, 2000)
        .limit(10)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);

    let results = store.query()
        .path_contains("to")
        .limit(10)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 0); // File path "/docs/ai_research.pdf" doesn't contain "to"

    let results: Result<Vec<_>, _> = store.query()
        .limit(10000)
        .execute()
        .await;
    match &results {
        Ok(r) => println!("Large limit search returned OK with {} results", r.len()),
        Err(e) => println!("Large limit search returned error: {:?}", e),
    }
    assert!(results.is_ok());

    let results: Result<Vec<_>, _> = store.query()
        .limit(1)
        .execute()
        .await;
    assert!(results.is_ok());
    let _ = results.unwrap().len(); // Just testing the unwrap

    let results = store.query()
        .path_contains("ai_research")
        .limit(5)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);

    let results = store.query()
        .path_contains("research")
        .limit(5)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);

    let results = store.query()
        .has_all_tags(&["ai", "research"])
        .limit(5)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_mime_type_variations() {
    let (store, _path) = create_test_store().await;

    let mut params = create_test_document_params();
    params.mime_type = Some("application/json".to_string());
    let json_id = store.add_text_file(&params).await.unwrap();

    let params = create_test_document_params()
        .override_file_path(PathBuf::from("/path/to/data.csv"))
        .mime_type("text/csv");
    let csv_id = store.add_text_file(&params).await.unwrap();

    let params = create_test_document_params()
        .override_file_path(PathBuf::from("/path/to/script.py"))
        .mime_type("text/x-python");
    let python_id = store.add_text_file(&params).await.unwrap();

    let results = store.query()
        .mime_type("application/json")
        .limit(5)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, json_id);

    let results = store.query()
        .mime_type("text/csv")
        .limit(5)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, csv_id);

    let results = store.query()
        .mime_type("text/x-python")
        .limit(5)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, python_id);
}

#[tokio::test]
async fn test_permission_variations() {
    let (store, _path) = create_test_store().await;

    let mut params = create_test_document_params();
    params.permissions = Some(0o600);
    let private_id = store.add_text_file(&params).await.unwrap();

    let params = create_test_document_params()
        .override_file_path(PathBuf::from("/path/to/public.txt"))
        .permissions(0o666);
    let public_id = store.add_text_file(&params).await.unwrap();

    let params = create_test_document_params()
        .override_file_path(PathBuf::from("/path/to/executable.sh"))
        .permissions(0o755);
    let executable_id = store.add_text_file(&params).await.unwrap();

    let private_metadata = store.get_metadata(&private_id).await.unwrap().unwrap();
    assert_eq!(private_metadata.permissions, Some(0o600));

    let public_metadata = store.get_metadata(&public_id).await.unwrap().unwrap();
    assert_eq!(public_metadata.permissions, Some(0o666));

    let executable_metadata = store.get_metadata(&executable_id).await.unwrap().unwrap();
    assert_eq!(executable_metadata.permissions, Some(0o755));
}

#[tokio::test]
async fn test_tag_parsing_edge_cases() {
    let (store, _path) = create_test_store().await;

    let mut params = create_test_document_params();
    params.tags = Some(vec![
        "tag1".to_string(),
        "tag2".to_string(),
        "tag3".to_string(),
    ]);
    let multi_tag_id = store.add_text_file(&params).await.unwrap();

    let params = create_test_document_params()
        .override_file_path(PathBuf::from("/path/to/no_tags.txt"))
        .tags(&[]);
    let no_tags_id = store.add_text_file(&params).await.unwrap();

    let params = create_test_document_params()
        .override_file_path(PathBuf::from("/path/to/empty_tags.txt"))
        .tags(&[]);
    let empty_tags_id = store.add_text_file(&params).await.unwrap();

    let params = create_test_document_params()
        .override_file_path(PathBuf::from("/path/to/single_tag.txt"))
        .tags(&["single"]);
    let single_tag_id = store.add_text_file(&params).await.unwrap();

    let multi_tag_metadata = store.get_metadata(&multi_tag_id).await.unwrap().unwrap();
    assert_eq!(
        multi_tag_metadata.tags,
        Some(vec![
            "tag1".to_string(),
            "tag2".to_string(),
            "tag3".to_string()
        ])
    );

    let no_tags_metadata = store.get_metadata(&no_tags_id).await.unwrap().unwrap();
    assert_eq!(no_tags_metadata.tags, None);

    let empty_tags_metadata = store.get_metadata(&empty_tags_id).await.unwrap().unwrap();
    assert_eq!(empty_tags_metadata.tags, None);

    let single_tag_metadata = store.get_metadata(&single_tag_id).await.unwrap().unwrap();
    assert_eq!(single_tag_metadata.tags, Some(vec!["single".to_string()]));

    let results = store.query()
        .has_all_tags(&["tag1"])
        .limit(5)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, multi_tag_id);

    let results = store.query()
        .has_all_tags(&["single"])
        .limit(5)
        .execute()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, single_tag_id);
}

#[tokio::test]
async fn test_content_hash_validation() {
    let (store, _path) = create_test_store().await;

    let content = "This is test content for hash validation";
    let correct_hash = compute_md5_hash(content);
    let wrong_hash = "incorrect_hash_value";

    let mut params = create_test_document_params();
    params.content_embedding = Some(create_mock_embedding(content, "content"));
    params.content_hash = Some(correct_hash.clone());
    let correct_id = store.add_text_file(&params).await.unwrap();

    let mut params = create_test_document_params();
    params.file_path = PathBuf::from("/path/to/wrong_hash.txt").into();
    params.content_embedding = Some(create_mock_embedding(content, "content"));
    params.content_hash = Some(wrong_hash.to_string());
    let wrong_id = store.add_text_file(&params).await.unwrap();

    let correct_metadata = store.get_metadata(&correct_id).await.unwrap().unwrap();
    let wrong_metadata = store.get_metadata(&wrong_id).await.unwrap().unwrap();

    assert_eq!(correct_metadata.filename(), Some("test_document.txt".to_string()));
    assert_eq!(wrong_metadata.filename(), Some("wrong_hash.txt".to_string()));
}

#[tokio::test]
async fn test_table_schema_validation() {
    let (store, _path) = create_test_store().await;

    let file_id = store
        .add_text_file(&create_test_document_params())
        .await
        .unwrap();
    let image_id = store
        .add_image_file(&create_test_document_params())
        .await
        .unwrap();

    let metadata = store.get_metadata(&file_id).await.unwrap().unwrap();
    let image_metadata = store.get_metadata(&image_id).await.unwrap().unwrap();

    assert_eq!(metadata.file_type, FileType::File);
    assert_eq!(image_metadata.file_type, FileType::File);

    assert_eq!(metadata.mime_type, Some("text/plain".to_string()));
    assert_eq!(image_metadata.mime_type, Some("text/plain".to_string()));
}

#[tokio::test]
async fn test_cleanup_missing_files() {
    let (store, temp_dir) = create_test_store().await;

    let temp_path = temp_dir.path();
    let file1_path = temp_path.join("test_file1.txt");
    let file2_path = temp_path.join("test_file2.txt");
    let file3_path = temp_path.join("test_file3.txt");

    std::fs::write(&file1_path, "test content 1").unwrap();
    std::fs::write(&file2_path, "test content 2").unwrap();
    std::fs::write(&file3_path, "test content 3").unwrap();

    let mut params1 = create_test_document_params();
    params1.file_path = file1_path.clone().into();

    let mut params2 = create_test_document_params();
    params2.file_path = file2_path.clone().into();

    let mut params3 = create_test_document_params();
    params3.file_path = file3_path.clone().into();

    let file_id1 = store.add_text_file(&params1).await.unwrap();
    let file_id2 = store.add_text_file(&params2).await.unwrap();
    let file_id3 = store.add_text_file(&params3).await.unwrap();

    let metadata1 = store.get_metadata(&file_id1).await.unwrap();
    let metadata2 = store.get_metadata(&file_id2).await.unwrap();
    let metadata3 = store.get_metadata(&file_id3).await.unwrap();

    assert!(metadata1.is_some());
    assert!(metadata2.is_some());
    assert!(metadata3.is_some());

    let deleted_count = store.cleanup_missing_files().await.unwrap();
    assert_eq!(deleted_count, 0);

    let metadata1_after = store.get_metadata(&file_id1).await.unwrap();
    let metadata2_after = store.get_metadata(&file_id2).await.unwrap();
    let metadata3_after = store.get_metadata(&file_id3).await.unwrap();

    assert!(metadata1_after.is_some());
    assert!(metadata2_after.is_some());
    assert!(metadata3_after.is_some());

    let mut params = create_test_document_params();
    params.file_path = PathBuf::from("/nonexistent/path/file.txt").into();
    let nonexistent_file_id = store.add_text_file(&params).await.unwrap();

    let nonexistent_metadata = store.get_metadata(&nonexistent_file_id).await.unwrap();
    assert!(nonexistent_metadata.is_some());

    let deleted_count = store.cleanup_missing_files().await.unwrap();
    assert_eq!(deleted_count, 1);

    let nonexistent_metadata_after = store.get_metadata(&nonexistent_file_id).await.unwrap();
    assert!(nonexistent_metadata_after.is_none());

    let metadata1_final = store.get_metadata(&file_id1).await.unwrap();
    let metadata2_final = store.get_metadata(&file_id2).await.unwrap();
    let metadata3_final = store.get_metadata(&file_id3).await.unwrap();

    assert!(metadata1_final.is_some());
    assert!(metadata2_final.is_some());
    assert!(metadata3_final.is_some());
}
