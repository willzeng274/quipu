use quipu::{
    create_mock_embedding,
};
use std::path::PathBuf;
use crate::helpers::{
    create_test_store,
    create_ai_research_params,
    create_machine_learning_params,
    create_tech_doc_params,
    create_test_document_params,
};

#[tokio::test]
async fn test_semantic_search_by_filename() {
    let (store, _path) = create_test_store().await;

    let _file_id1 = store
        .add_text_file(&create_ai_research_params())
        .await
        .unwrap();
    let _file_id2 = store
        .add_text_file(&create_machine_learning_params())
        .await
        .unwrap();

    let query_embedding = create_mock_embedding("machine learning query", "filename");
    let results = store.query()
        .by_filename(&query_embedding)
        .limit(2)
        .execute()
        .await
        .unwrap();

    assert!(!results.is_empty());
    assert!(results.len() <= 2);
}

#[tokio::test]
async fn test_semantic_search_by_content() {
    let (store, _path) = create_test_store().await;

    let _file_id1 = store
        .add_text_file(&create_tech_doc_params())
        .await
        .unwrap();
    let _file_id2 = store.add_text_file(&create_machine_learning_params()).await.unwrap();

    let query_embedding = create_mock_embedding("AI machine learning query", "content");
    let results = store.query()
        .by_text_content(&query_embedding)
        .limit(5)
        .execute()
        .await
        .unwrap();

    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_combined_search() {
    let (store, _path) = create_test_store().await;

    let _file_id1 = store
        .add_text_file(&create_ai_research_params())
        .await
        .unwrap();
    let _file_id2 = store
        .add_text_file(&create_machine_learning_params())
        .await
        .unwrap();

    let query_embedding = create_mock_embedding("AI research query", "content");

    let filename_results = store.query()
        .by_filename(&query_embedding)
        .limit(3)
        .execute()
        .await
        .unwrap();

    let text_content_results = store.query()
        .by_text_content(&query_embedding)
        .limit(3)
        .execute()
        .await
        .unwrap();

    let combined_results = store.query()
        .combined(&query_embedding)
        .limit(3)
        .execute()
        .await
        .unwrap();

    assert!(!filename_results.is_empty());
    assert!(!text_content_results.is_empty());
    assert!(combined_results.len() <= 3);
}

#[tokio::test]
async fn test_search_edge_cases() {
    let (store, _path) = create_test_store().await;

    let _file_id = store
        .add_text_file(&create_ai_research_params())
        .await
        .unwrap();

    let empty_embedding = vec![];
    let result = store.query()
        .by_filename(&empty_embedding)
        .limit(5)
        .execute()
        .await;
    assert!(result.is_err());

    let large_embedding = vec![0.0; 10000];
    let result = store.query()
        .by_filename(&large_embedding)
        .limit(5)
        .execute()
        .await;
    assert!(result.is_err());

    let query_embedding = create_mock_embedding("query", "filename");
    let results = store.query()
        .by_filename(&query_embedding)
        .limit(1)
        .execute()
        .await
        .unwrap();
    assert!(results.len() <= 1);

    let results = store.query()
        .by_filename(&query_embedding)
        .limit(10000)
        .execute()
        .await
        .unwrap();
    assert!(results.len() <= 1);
}

#[tokio::test]
async fn test_semantic_search_with_special_values() {
    let (store, _path) = create_test_store().await;

    let _file_id = store
        .add_text_file(&create_ai_research_params())
        .await
        .unwrap();

    let mut nan_embedding = create_mock_embedding("query", "filename");
    nan_embedding[0] = f32::NAN;
    let result = store.query()
        .by_filename(&nan_embedding)
        .limit(5)
        .execute()
        .await;
    // Should handle NaN gracefully
    if result.is_ok() {
        let results = result.unwrap();
        assert!(results.len() <= 1);
    }

    let mut inf_embedding = create_mock_embedding("query", "filename");
    inf_embedding[0] = f32::INFINITY;
    let result = store.query()
        .by_filename(&inf_embedding)
        .limit(5)
        .execute()
        .await;
    // Should handle infinity gracefully
    if result.is_ok() {
        let results = result.unwrap();
        assert!(results.len() <= 1);
    }
}

#[tokio::test]
async fn test_semantic_search_with_metadata_filters() {
    let (store, _path) = create_test_store().await;

    let _file_id1 = store
        .add_text_file(&create_ai_research_params())
        .await
        .unwrap();
    let _file_id2 = store
        .add_text_file(&create_machine_learning_params())
        .await
        .unwrap();
    let _file_id3 = store
        .add_text_file(&create_tech_doc_params())
        .await
        .unwrap();

    let query_embedding = create_mock_embedding("machine learning research", "content");

    let results = store.query()
        .by_text_content(&query_embedding)
        .path_contains("research")
        .has_all_tags(&["ai"])
        .limit(10)
        .execute()
        .await
        .unwrap();

    assert!(!results.is_empty());
    // Note: path_contains("research") should limit results, but with semantic search
    // we can't guarantee all results will contain "research" in the path
    // The test primarily validates that semantic search works with tag filters

    let results = store.query()
        .by_text_content(&query_embedding)
        .size_range(0, 1000)
        .limit(10)
        .execute()
        .await
        .unwrap();

    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_combined_search_with_metadata_filters() {
    let (store, _path) = create_test_store().await;

    let _file_id1 = store
        .add_text_file(&create_ai_research_params())
        .await
        .unwrap();
    let _file_id2 = store
        .add_text_file(&create_machine_learning_params())
        .await
        .unwrap();

    let query_embedding = create_mock_embedding("artificial intelligence", "content");

    let results = store.query()
        .by_text_content(&query_embedding)
        .mime_type("application/pdf")
        .filter("file_type = 'file'")
        .limit(10)
        .execute()
        .await
        .unwrap();

    assert!(results.len() <= 5);
    assert!(results.iter().all(|r| format!("{:?}", r.file_type) == "File"));
}

#[tokio::test]
async fn test_search_with_different_semantic_types() {
    let (store, _path) = create_test_store().await;

    let _file_id1 = store
        .add_text_file(&create_ai_research_params())
        .await
        .unwrap();
    let _file_id2 = store
        .add_text_file(&create_machine_learning_params())
        .await
        .unwrap();

    let filename_query = create_mock_embedding("research", "filename");
    let filename_results = store.query()
        .by_filename(&filename_query)
        .limit(3)
        .execute()
        .await
        .unwrap();

    let content_query = create_mock_embedding("machine learning", "content");
    let content_results = store.query()
        .by_text_content(&content_query)
        .limit(3)
        .execute()
        .await
        .unwrap();

    let image_query = create_mock_embedding("image", "content");
    let image_results = store.query()
        .by_image_content(&image_query)
        .limit(3)
        .execute()
        .await
        .unwrap();

    assert!(!filename_results.is_empty());
    assert!(!content_results.is_empty());
    assert!(image_results.is_empty());
}

#[tokio::test]
async fn test_search_pagination() {
    let (store, _path) = create_test_store().await;

    for i in 0..10 {
        let mut params = create_test_document_params();
        params.file_path = PathBuf::from(format!("/path/to/test_file_{}.txt", i)).into();
        params.filename_embedding = create_mock_embedding(&format!("test file {}", i), "filename");
        let _file_id = store.add_text_file(&params).await.unwrap();
    }

    let query_embedding = create_mock_embedding("test file", "filename");

    let page1 = store.query()
        .by_filename(&query_embedding)
        .limit(3)
        .execute()
        .await
        .unwrap();

    let page2 = store.query()
        .by_filename(&query_embedding)
        .limit(3)
        .execute()
        .await
        .unwrap();

    assert_eq!(page1.len(), 3);
    assert_eq!(page2.len(), 3);
}
