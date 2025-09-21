#[cfg(windows)]
mod windows_utf16_integration_tests {
    use super::*;
    use std::fs;
    use quipu::{FilePath};
    use std::path::PathBuf;
    use crate::helpers::{
        create_test_store,
    };

    fn create_utf16_path(filename: &str) -> PathBuf {
        let utf16_chars: Vec<u16> = vec![
            0x0043, 0x003A, 0x005C, 0x0074, 0x0065, 0x006D, 0x0070, 0x005C, // C:\temp\
        ].into_iter().chain(
            filename.encode_utf16().collect::<Vec<u16>>()
        ).collect();

        let os_string = unsafe { OsString::from_encoded_bytes_unchecked(utf16_chars) };
        PathBuf::from(os_string)
    }

    fn create_utf16_file_params(utf16_path: PathBuf, content: &str) -> AddFile {
        let filename = utf16_path.file_name().unwrap().to_string_lossy();
        let filename_embedding = create_mock_embedding(&filename, "filename");
        let content_embedding = create_mock_embedding(content, "content");

        AddFile::new(utf16_path, FileType::File, filename_embedding)
            .content_embedding(content_embedding)
            .content_hash(content)
            .mime_type("text/plain")
            .tags(&["utf16", "test"])
            .file_size(content.len() as u64)
            .permissions(0o644)
    }

    #[tokio::test]
    async fn test_utf16_file_add_and_retrieve() {
        let (store, temp_dir) = create_test_store().await;
        let temp_path = temp_dir.path();

        let utf16_filename = "utf16_test.txt";
        let utf16_path = create_utf16_path(utf16_filename);
        let content = "This is UTF-16 file content";

        fs::write(&utf16_path, content).unwrap();

        let params = create_utf16_file_params(utf16_path.clone(), content);
        let file_id = store.add_text_file(&params).await.unwrap();

        let metadata = store.get_metadata(&file_id).await.unwrap().unwrap();
        assert_eq!(metadata.file_type, FileType::File);
        assert!(metadata.filename().unwrap().contains(utf16_filename));

        fs::remove_file(&utf16_path).unwrap();
    }

    #[tokio::test]
    async fn test_utf16_file_search_by_filename() {
        let (store, _temp_dir) = create_test_store().await;

        let utf16_filename = "search_utf16.txt";
        let utf16_path = create_utf16_path(utf16_filename);
        let content = "Search test content";

        fs::write(&utf16_path, content).unwrap();

        let params = create_utf16_file_params(utf16_path.clone(), content);
        let _file_id = store.add_text_file(&params).await.unwrap();

        let query_embedding = create_mock_embedding(utf16_filename, "filename");
        let results = store.search_by_filename(&query_embedding, 5).await.unwrap();

        assert!(!results.is_empty());
        assert!(results[0].filename().unwrap().contains(utf16_filename));

        fs::remove_file(&utf16_path).unwrap();
    }

    #[tokio::test]
    async fn test_utf16_file_search_by_content() {
        let (store, _temp_dir) = create_test_store().await;

        let utf16_filename = "content_utf16.txt";
        let utf16_path = create_utf16_path(utf16_filename);
        let content = "UTF-16 content search test";

        fs::write(&utf16_path, content).unwrap();

        let params = create_utf16_file_params(utf16_path.clone(), content);
        let _file_id = store.add_text_file(&params).await.unwrap();

        let query_embedding = create_mock_embedding("UTF-16 content", "content");
        let results = store.search_by_text_content(&query_embedding, 5).await.unwrap();

        assert!(!results.is_empty());

        fs::remove_file(&utf16_path).unwrap();
    }

    #[tokio::test]
    async fn test_utf16_combined_search() {
        let (store, _temp_dir) = create_test_store().await;

        let utf16_filename = "combined_utf16.txt";
        let utf16_path = create_utf16_path(utf16_filename);
        let content = "Combined search UTF-16 content";

        fs::write(&utf16_path, content).unwrap();

        let params = create_utf16_file_params(utf16_path.clone(), content);
        let _file_id = store.add_text_file(&params).await.unwrap();

        let query_embedding = create_mock_embedding("combined_utf16", "filename");
        let combined_results = store.search_combined(&query_embedding, 5).await.unwrap();

        assert!(!combined_results.is_empty());

        fs::remove_file(&utf16_path).unwrap();
    }

    #[tokio::test]
    async fn test_utf16_file_delete_operation() {
        let (store, _temp_dir) = create_test_store().await;

        let utf16_filename = "delete_utf16.txt";
        let utf16_path = create_utf16_path(utf16_filename);
        let content = "Content to be deleted";

        fs::write(&utf16_path, content).unwrap();

        let params = create_utf16_file_params(utf16_path.clone(), content);
        let _file_id = store.add_text_file(&params).await.unwrap();

        let metadata_before = store.get_metadata_by_path(&utf16_path).await.unwrap();
        assert!(metadata_before.is_some());

        store.delete_file(&utf16_path).await.unwrap();

        let metadata_after = store.get_metadata_by_path(&utf16_path).await.unwrap();
        assert!(metadata_after.is_none());

        fs::remove_file(&utf16_path).unwrap();
    }

    #[tokio::test]
    async fn test_utf16_file_move_operation() {
        let (store, _temp_dir) = create_test_store().await;

        let original_filename = "move_original_utf16.txt";
        let new_filename = "move_new_utf16.txt";
        let original_path = create_utf16_path(original_filename);
        let new_path = create_utf16_path(new_filename);
        let content = "Content to be moved";

        fs::write(&original_path, content).unwrap();

        let params = create_utf16_file_params(original_path.clone(), content);
        let _file_id = store.add_text_file(&params).await.unwrap();

        store.move_file(&original_path, new_path.clone()).await.unwrap();

        let metadata_after = store.get_metadata_by_path(&new_path).await.unwrap();
        assert!(metadata_after.is_some());
        assert!(metadata_after.unwrap().filename().unwrap().contains(new_filename));

        let old_metadata = store.get_metadata_by_path(&original_path).await.unwrap();
        assert!(old_metadata.is_none());

        fs::remove_file(&new_path).unwrap();
    }

    #[tokio::test]
    async fn test_utf16_file_reindex_operation() {
        let (store, _temp_dir) = create_test_store().await;

        let utf16_filename = "reindex_utf16.txt";
        let utf16_path = create_utf16_path(utf16_filename);
        let original_content = "Original content";

        fs::write(&utf16_path, original_content).unwrap();

        let params = create_utf16_file_params(utf16_path.clone(), original_content);
        let _file_id = store.add_text_file(&params).await.unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        let new_content = "Updated content after reindex";
        fs::write(&utf16_path, new_content).unwrap();

        store.reindex_file(&utf16_path).await.unwrap();

        let metadata_after = store.get_metadata_by_path(&utf16_path).await.unwrap();
        assert!(metadata_after.is_some());
        assert_eq!(metadata_after.unwrap().file_size, new_content.len() as u64);

        fs::remove_file(&utf16_path).unwrap();
    }

    #[tokio::test]
    async fn test_utf16_file_update_content() {
        let (store, _temp_dir) = create_test_store().await;

        let utf16_filename = "update_utf16.txt";
        let utf16_path = create_utf16_path(utf16_filename);
        let original_content = "Original UTF-16 content";

        fs::write(&utf16_path, original_content).unwrap();

        let params = create_utf16_file_params(utf16_path.clone(), original_content);
        let _file_id = store.add_text_file(&params).await.unwrap();

        let new_content = "Updated UTF-16 content with more text";
        let new_embedding = create_mock_embedding(new_content, "content");
        let new_content_hash = compute_md5_hash(new_content);

        store.update_text_file_content(
            &utf16_path,
            new_content,
            new_embedding.clone(),
            &new_content_hash
        ).await.unwrap();

        let metadata_after = store.get_metadata_by_path(&utf16_path).await.unwrap();
        assert!(metadata_after.is_some());
        let metadata = metadata_after.unwrap();
        assert_eq!(metadata.file_size, new_content.len() as u64);

        let search_results = store.search_by_text_content(&new_embedding, 5).await.unwrap();
        assert!(!search_results.is_empty());

        fs::remove_file(&utf16_path).unwrap();
    }

    #[tokio::test]
    async fn test_utf16_multiple_files_operations() {
        let (store, _temp_dir) = create_test_store().await;

        let files = vec![
            ("multi1_utf16.txt", "Content for file 1"),
            ("multi2_utf16.txt", "Content for file 2"),
            ("multi3_utf16.txt", "Content for file 3"),
        ];

        let mut file_ids = vec![];

        for (filename, content) in &files {
            let utf16_path = create_utf16_path(filename);
            fs::write(&utf16_path, content).unwrap();

            let params = create_utf16_file_params(utf16_path.clone(), content);
            let file_id = store.add_text_file(&params).await.unwrap();
            file_ids.push((file_id, utf16_path));
        }

        assert_eq!(file_ids.len(), 3);

        for (file_id, utf16_path) in &file_ids {
            let metadata = store.get_metadata(file_id).await.unwrap();
            assert!(metadata.is_some());
            fs::remove_file(utf16_path).unwrap();
        }
    }

    #[tokio::test]
    async fn test_utf16_file_tags_operations() {
        let (store, _temp_dir) = create_test_store().await;

        let utf16_filename = "tags_utf16.txt";
        let utf16_path = create_utf16_path(utf16_filename);
        let content = "Content with UTF-16 tags";

        fs::write(&utf16_path, content).unwrap();

        let mut params = create_utf16_file_params(utf16_path.clone(), content);
        params.tags = Some(vec!["utf16".to_string(), "special".to_string()]);

        let file_id = store.add_text_file(&params).await.unwrap();

        let new_tags = vec!["updated".to_string(), "production".to_string(), "utf16".to_string()];
        store.update_tags(&utf16_path, new_tags.clone()).await.unwrap();

        let metadata_after = store.get_metadata_by_path(&utf16_path).await.unwrap();
        assert!(metadata_after.is_some());
        let metadata = metadata_after.unwrap();
        assert_eq!(metadata.tags, Some(new_tags));

        let search_results = store.search_by_tags(&["utf16".to_string()], 5).await.unwrap();
        assert!(!search_results.is_empty());
        assert_eq!(search_results[0].id, file_id);

        fs::remove_file(&utf16_path).unwrap();
    }
}