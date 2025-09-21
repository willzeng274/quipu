use quipu::MetadataUpdate;
use std::path::PathBuf;

#[test]
fn test_metadata_update_new() {
    let update = MetadataUpdate::new();

    assert!(update.file_path.is_none());
    assert!(update.mime_type.is_none());
    assert!(update.permissions.is_none());
    assert!(update.file_size.is_none());
    assert!(update.modified_time.is_none());
    assert!(update.accessed_time.is_none());
    assert!(update.created_time.is_none());
    assert!(update.tags.is_none());
}

#[test]
fn test_metadata_update_builder_pattern() {
    let update = MetadataUpdate::new()
        .file_path(PathBuf::from("/new/path/file.txt"))
        .mime_type("application/json")
        .permissions(0o644)
        .file_size(2048)
        .modified_time(1234567890)
        .accessed_time(1234567891)
        .created_time(1234567889)
        .tags(&["updated", "important"]);

    assert_eq!(PathBuf::from(update.file_path.unwrap()), PathBuf::from("/new/path/file.txt"));
    assert_eq!(update.mime_type, Some("application/json".to_string()));
    assert_eq!(update.permissions, Some(0o644));
    assert_eq!(update.file_size, Some(2048));
    assert_eq!(update.modified_time, Some(1234567890));
    assert_eq!(update.accessed_time, Some(1234567891));
    assert_eq!(update.created_time, Some(1234567889));
    assert_eq!(update.tags, Some(vec!["updated".to_string(), "important".to_string()]));
}

#[test]
fn test_metadata_update_partial() {
    let update = MetadataUpdate::new()
        .file_size(1024)
        .modified_time(1234567890);

    assert!(update.file_path.is_none());
    assert!(update.mime_type.is_none());
    assert!(update.permissions.is_none());
    assert_eq!(update.file_size, Some(1024));
    assert_eq!(update.modified_time, Some(1234567890));
    assert!(update.accessed_time.is_none());
    assert!(update.created_time.is_none());
    assert!(update.tags.is_none());
}

#[test]
fn test_metadata_update_tags() {
    let update = MetadataUpdate::new()
        .tags(&["tag1", "tag2", "tag3"]);

    assert_eq!(update.tags, Some(vec!["tag1".to_string(), "tag2".to_string(), "tag3".to_string()]));
}

#[test]
fn test_metadata_update_chaining() {
    let update = MetadataUpdate::new()
        .file_path(PathBuf::from("/path/file.txt"))
        .mime_type("text/plain")
        .permissions(0o644)
        .file_size(1024)
        .modified_time(1234567890)
        .accessed_time(1234567891)
        .created_time(1234567889)
        .tags(&["test"]);

    // Verify all fields are set correctly
    assert_eq!(PathBuf::from(update.file_path.unwrap()), PathBuf::from("/path/file.txt"));
    assert_eq!(update.mime_type, Some("text/plain".to_string()));
    assert_eq!(update.permissions, Some(0o644));
    assert_eq!(update.file_size, Some(1024));
    assert_eq!(update.modified_time, Some(1234567890));
    assert_eq!(update.accessed_time, Some(1234567891));
    assert_eq!(update.created_time, Some(1234567889));
    assert_eq!(update.tags, Some(vec!["test".to_string()]));
}
